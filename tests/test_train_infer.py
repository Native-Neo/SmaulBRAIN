"""End-to-end: real training run, inference run, fine-tune modes, quantize CLI."""

import sys, os, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SmaulBrainConfig
from infer import generate
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import batch_from_seqs, evaluate_loss, run_training, train_step

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _model(**kw):
    torch.manual_seed(0)
    base = dict(d_model=32, n_heads=4, num_experts=4, top_k=2, expert_hidden=64,
                max_depth=2, context_length=24, expert_lr=3e-2)
    base.update(kw)
    cfg = SmaulBrainConfig(**base)
    return SmaulBrainModel(cfg), SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr)), cfg


def _seqs(n=16, seed=1):
    import random
    rng = random.Random(seed)
    return [[rng.randrange(256) for _ in range(30)] for _ in range(n)]


def test_training_loss_decreases_on_real_run():
    m, opt, cfg = _model()
    res = run_training(m, opt, cfg, _seqs(), steps=10, batch_size=2)
    first, final = res["history"][0]["loss"], res["final_loss"]
    assert final < first, f"{first} -> {final}"
    m.pager.close()


def test_inference_generates_bytes():
    m, _, _ = _model()
    res = generate(m, [104, 105], max_new=6, temperature=0.0)
    assert len(res["ids"]) == 8 and all(0 <= i < 256 for i in res["ids"][2:])
    m.pager.close()


def test_finetune_modes_move_right_groups():
    for mode in ("entire", "trunk", "experts", "selected", "new"):
        m, opt, cfg = _model()
        b = torch.randint(0, 256, (2, cfg.context_length + 1))
        kw = {}
        if mode == "selected":
            kw["selected"] = [m.pool.order[0]]
        if mode == "new":
            kw["new_since_step"] = 0
        s = train_step(m, opt, cfg, b[:, :cfg.context_length],
                       b[:, 1:], step=0, mode=mode, **kw)
        stepped = s["stepped_experts"]
        if mode == "trunk":
            assert stepped == []
        else:
            assert len(stepped) > 0
        if mode == "selected":
            assert set(stepped) <= set(kw["selected"])
        m.pager.close()


def test_cli_train_then_infer_then_quantize(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    r = subprocess.run(
        [sys.executable, "cli.py", "--d-model", "32", "--n-heads", "4",
         "--experts", "2", "--expert-size", "64", "--active-experts", "1",
         "--max-depth", "1", "--context-length", "24", "--ckpt", ckpt,
         "train", "--steps", "3", "--batch", "2"], capture_output=True,
        text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["final_loss"] > 0
    r = subprocess.run(
        [sys.executable, "cli.py", "--d-model", "32", "--n-heads", "4",
         "--experts", "2", "--expert-size", "64", "--active-experts", "1",
         "--max-depth", "1", "--context-length", "24", "--ckpt", ckpt,
         "infer", "--prompt", "hi", "--max-new", "4", "--temperature", "0"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [sys.executable, "cli.py", "--ckpt", ckpt, "quantize", "--to", "fp8"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


def test_streaming_inference_matches_full_causal_pass():
    m, _, _ = _model()
    ids = torch.tensor([[10, 20, 30, 40, 50, 60]])
    full, _ = m.forward_infer_stateful(ids)
    states = m.new_infer_state(1)
    parts = []
    for token in ids[0]:
        out, states = m.forward_infer_step(token.view(1), states)
        parts.append(out["logits"])
    streamed = torch.cat(parts, dim=1)
    assert torch.allclose(full["logits"], streamed, atol=2e-5, rtol=2e-5)
    m.pager.close()


def test_special_token_ids_train_without_index_errors():
    from bytes import with_bos, with_eos
    m, opt, cfg = _model()
    assert cfg.vocab_size >= 260  # specials inside model bounds
    seq = with_eos(with_bos([10, 20, 30, 40]))
    b = batch_from_seqs([seq], context=cfg.context_length)
    s = train_step(m, opt, cfg, b[:, :cfg.context_length], b[:, 1:], step=0)
    assert torch.isfinite(torch.tensor(s["loss"]))
    m.pager.close()


def test_eval_loss_invariant_to_pad_tail_length():
    m, _, cfg = _model()
    seq = [10, 20, 30, 40, 50, 60]
    got = []
    for ctx in (8, 12, 20):
        b = batch_from_seqs([seq], context=ctx)
        got.append(evaluate_loss(m, b, ctx)["loss"])
    assert got[0] == got[1] == got[2]  # pads contribute nothing, norm is valid-only
    m.pager.close()


def test_train_loss_invariant_to_pad_tail_without_capacity_pressure():
    # Capacity binds nothing here (factor 1000), so per-position outputs are
    # prefix-determined and the valid-normalized loss must match exactly.
    m, _, cfg = _model(capacity_factor=1000.0)
    seq = [10, 20, 30, 40, 50, 60]
    got = []
    for ctx in (8, 12, 20):
        b = batch_from_seqs([seq], context=ctx)
        with torch.no_grad():
            got.append(float(m(b[:, :ctx], b[:, 1:], step=0)["loss"]))
    assert got[0] == got[1] == got[2]
    m.pager.close()
