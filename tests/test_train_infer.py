"""End-to-end: real training run, inference run, fine-tune modes, quantize CLI."""

import sys, os, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
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


def test_cli_explicit_flag_beats_full_preset():
    from cli import build_parser, config_from_args
    args = build_parser().parse_args(["--full", "--max-depth", "3", "train"])
    cfg = config_from_args(args)
    assert cfg.max_depth == 3  # explicit tiny-equal value wins over preset
    assert cfg.d_model == 512  # silent flags follow the preset
    tiny = config_from_args(build_parser().parse_args(["train"]))
    assert (tiny.d_model, tiny.max_depth) == (64, 3)


def test_cli_paging_aliases_and_chunk_size():
    from cli import build_parser, config_from_args
    cfg = config_from_args(build_parser().parse_args(
        ["--paging-method", "r2vr", "--attention-chunk-size", "64", "train"]))
    assert cfg.paging_method == "R2VR"
    assert cfg.attention_chunk_size == 64


def test_generate_rejects_out_of_range_prompt_ids():
    m, _, _ = _model()
    with pytest.raises(ValueError):
        generate(m, [10, 99999], max_new=2)
    with pytest.raises(ValueError):
        generate(m, [10, -1], max_new=2)
    m.pager.close()


def test_sampler_rejects_nonsense_hyperparams():
    from infer import sample_next
    l = torch.zeros(260)
    for kw in (dict(temperature=float("nan")), dict(top_p=0.0), dict(top_p=1.5)):
        with pytest.raises(ValueError):
            sample_next(l, **kw)


def test_top_p_keeps_true_nucleus():
    from infer import sample_next
    torch.manual_seed(0)
    l = torch.tensor([10.0, 9.0, 0.0, -100.0])
    seen = {sample_next(l, temperature=1.0, top_p=0.9) for _ in range(50)}
    assert seen and seen <= {0, 1}  # ~all mass on the top two, tail excluded


def test_sampler_rejects_bad_top_k_and_temperature():
    from infer import sample_next
    l = torch.zeros(260)
    bad = [
        dict(temperature=-1.0),
        dict(temperature=float("inf")),
        dict(temperature="hot"),
        dict(temperature=True),
        dict(top_k=-1),
        dict(top_k=1.5),
        dict(top_k="2"),
        dict(top_k=True),
        dict(top_p=float("nan")),
        dict(top_p="x"),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            sample_next(l, **kw)


def test_generate_validates_sampler_upfront():
    m, _, _ = _model()
    with pytest.raises(ValueError):
        generate(m, [10, 20], max_new=2, top_k=-1)
    with pytest.raises(ValueError):
        generate(m, [10, 20], max_new=2, temperature=-0.5)
    m.pager.close()


def test_generate_seeded_deterministic_and_text_matches_ids():
    from bytes import decode_text
    m, _, _ = _model()
    a = generate(m, [104, 105], max_new=8, temperature=0.7, seed=7)
    b = generate(m, [104, 105], max_new=8, temperature=0.7, seed=7)
    assert a["ids"] == b["ids"] and a["text"] == b["text"]
    assert a["text"] == decode_text(a["ids"][2:])  # continuation-only decode
    # Empty prompt falls back to DEFAULT_PROMPT_ID; max_new=0 echoes prompt.
    e = generate(m, [], max_new=4, temperature=0.0)
    assert len(e["ids"]) == 5 and e["text"] == decode_text(e["ids"][1:])
    z = generate(m, [104, 105], max_new=0, temperature=0.0)
    assert z["ids"] == [104, 105] and z["text"] == "" and z["depths"] == []
    m.pager.close()


def test_cli_maps_every_config_field():
    from cli import build_parser, config_from_args
    from config import SmaulBrainConfig
    import dataclasses
    fields = {f.name for f in dataclasses.fields(SmaulBrainConfig)} - {"active_experts"}
    args = build_parser().parse_args(["train"])
    cfg = config_from_args(args)
    for name in sorted(fields):
        assert hasattr(cfg, name), name
    # Ponder / balance / optimizer / precision / norm / vocab / growth knobs.
    args = build_parser().parse_args([
        "--halt-prior", "0.2", "--ponder-beta", "0.03",
        "--moe-balance-weight", "0.05", "--router-lr-mult", "0.5",
        "--weight-decay", "0.02", "--grad-clip", "0.5",
        "--beta-m", "0.8", "--beta-v", "0.99", "--epsilon", "1e-7",
        "--fp8-tile", "32", "--state-dtype", "fp32",
        "--rmsnorm-eps", "1e-5", "--vocab-size", "260",
        "--prune-survival-steps", "10", "--prune-min-usage", "0.01",
        "--max-new-experts", "3", "--grow-every-default", "50",
        "train"])
    cfg = config_from_args(args)
    assert (cfg.halt_prior, cfg.ponder_beta, cfg.moe_balance_weight) == (0.2, 0.03, 0.05)
    assert (cfg.router_lr_mult, cfg.weight_decay, cfg.grad_clip) == (0.5, 0.02, 0.5)
    assert (cfg.beta_m, cfg.beta_v, cfg.epsilon) == (0.8, 0.99, 1e-7)
    assert (cfg.fp8_tile, cfg.state_dtype, cfg.rmsnorm_eps) == (32, "fp32", 1e-5)
    assert (cfg.prune_survival_steps, cfg.prune_min_usage, cfg.max_new_experts) == (10, 0.01, 3)
    assert cfg.grow_every == 50 and cfg.vocab_size == 260


def test_cli_shape_mismatch_aborts_before_build(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    base = ["--d-model", "32", "--n-heads", "4", "--experts", "2",
            "--expert-size", "64", "--active-experts", "1",
            "--max-depth", "1", "--context-length", "24",
            "--ckpt", ckpt]
    r = subprocess.run([sys.executable, "cli.py", *base, "train",
                        "--steps", "2", "--batch", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    bad = [b if b != "32" else "48" for b in base]
    r = subprocess.run([sys.executable, "cli.py", *bad, "train", "--steps", "1"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2
    assert "topology mismatch" in r.stderr and "d_model" in r.stderr


def test_cli_runtime_mismatch_warns_and_checkpoint_wins(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    base = ["--d-model", "32", "--n-heads", "4", "--experts", "2",
            "--expert-size", "64", "--active-experts", "1",
            "--max-depth", "1", "--context-length", "24",
            "--ckpt", ckpt]
    r = subprocess.run([sys.executable, "cli.py", *base, "train",
                        "--steps", "2", "--batch", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "cli.py", *base, "--ram-cache", "999",
                        "--ckpt", ckpt, "report"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "ignoring --ram-cache" in r.stderr
    body = json.loads(r.stdout)
    assert body["logical_params"] == body["total_params"]
    assert body["unique_params"] == body["total_params"]
    assert set(("paging", "resident")) <= set(body)


def test_cli_quantize_and_train_grow_every_schema(tmp_path):
    from cli import build_parser, config_from_args, resolve_train_grow_every
    ckpt = str(tmp_path / "ckpt")
    base = ["--d-model", "32", "--n-heads", "4", "--experts", "2",
            "--expert-size", "64", "--active-experts", "1",
            "--max-depth", "1", "--context-length", "24",
            "--ckpt", ckpt]
    r = subprocess.run([sys.executable, "cli.py", *base, "train",
                        "--steps", "2", "--batch", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "cli.py", "--ckpt", ckpt,
                        "quantize", "--to", "fp8"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    body = json.loads(r.stdout)
    assert body["to"] == "fp8" and body["converted_experts"] == 2
    assert body["bytes_before"] > 0 and body["bytes_after"] > 0
    # Train schedule falls back to config.grow_every when omitted.
    cfg = config_from_args(build_parser().parse_args(["train"]))
    assert resolve_train_grow_every(build_parser().parse_args(["train"]), cfg) == cfg.grow_every == 200
    assert resolve_train_grow_every(build_parser().parse_args(["train", "--grow-every", "0"]), cfg) == 0


# --- issue #56: special-token generation contract (append-only) ---

def _stub_model(plan, vocab_size=260, logits_vocab=None):
    """Minimal generate() double: scripted argmax ids, no training."""
    import types
    lv = logits_vocab if logits_vocab is not None else vocab_size
    state = {"i": 0}

    def _out_for(x):
        want = plan[min(state["i"], len(plan) - 1)]
        state["i"] += 1
        t = x.shape[1] if x.ndim == 2 else 1
        l = torch.full((1, t, lv), -10.0)
        if 0 <= want < lv:
            l[:, -1, want] = 10.0
        return {"logits": l, "depths": torch.zeros(1, t)}, []

    m = types.SimpleNamespace()
    m.training = False
    m.eval = lambda: setattr(m, "training", False)
    m.train = lambda: setattr(m, "training", True)
    m.embed = types.SimpleNamespace(
        weight=types.SimpleNamespace(device=torch.device("cpu")))
    m.cfg = types.SimpleNamespace(vocab_size=vocab_size, context_length=24)
    m.forward_infer_stateful = lambda x, attn_states=None, step=0: _out_for(x)
    # Step reuses _out_for (shares the script counter).
    m.forward_infer_step = lambda x, s, step=0: _out_for(x)
    m.pager = types.SimpleNamespace(
        stats=types.SimpleNamespace(to_dict=lambda: {}))
    return m


def test_generate_stops_on_eos_by_default():
    from bytes import EOS_ID, decode_text
    m = _stub_model([EOS_ID])
    res = generate(m, [104, 105], max_new=8, temperature=0.0)
    assert res["ids"][-1] == EOS_ID and len(res["ids"]) == 3
    assert len(res["depths"]) == 1  # EOS keeps its depth entry
    assert res["text"] == decode_text(res["ids"][2:]) == ""  # EOS decodes to no text


def test_generate_stop_on_eos_false_runs_full_length():
    from bytes import EOS_ID
    m = _stub_model([EOS_ID])
    res = generate(m, [104, 105], max_new=4, temperature=0.0, stop_on_eos=False)
    assert res["ids"][2:] == [EOS_ID] * 4 and len(res["depths"]) == 4


def test_generate_never_samples_pad_or_bos_greedy():
    from bytes import PAD_ID, BOS_ID
    for banned in (PAD_ID, BOS_ID):
        m = _stub_model([banned])  # stub wants the banned id on top
        res = generate(m, [104, 105], max_new=1, temperature=0.0)
        assert res["ids"][-1] != banned  # masked before argmax
        assert 0 <= res["ids"][-1] < 256  # falls back to a real byte


def test_sample_next_forbidden_masks_greedy_and_validates_vocab():
    from infer import sample_next
    from bytes import PAD_ID, BOS_ID
    l = torch.full((260,), -10.0)
    l[PAD_ID] = 100.0
    l[65] = 10.0
    assert sample_next(l, temperature=0.0, forbidden_ids=[PAD_ID, BOS_ID]) == 65
    assert sample_next(l, temperature=0.0) == PAD_ID  # no mask: old path intact
    with pytest.raises(ValueError):
        sample_next(torch.zeros(260), temperature=0.0, vocab_size=259)
    assert sample_next(torch.zeros(4), temperature=0.0, vocab_size=4) == 0
    with pytest.raises(ValueError):
        sample_next(torch.zeros(4), temperature=0.0, vocab_size=260)
    # Out-of-range forbiddens are ignored so tiny logits still work.
    assert sample_next(torch.zeros(4), temperature=0.0,
                       forbidden_ids=[99999]) == 0


def test_generate_rejects_logits_vocab_mismatch():
    from bytes import EOS_ID
    m = _stub_model([65], vocab_size=260, logits_vocab=4)
    with pytest.raises(ValueError):
        generate(m, [104, 105], max_new=2, temperature=0.0)


def test_generate_byte_zero_is_ordinary_data():
    m = _stub_model([0])
    res = generate(m, [0, 65], max_new=1, temperature=0.0)  # NUL prompt ok
    assert res["ids"][:2] == [0, 65] and res["ids"][-1] == 0  # NUL sampled ok
    assert res["text"] == "\x00"  # NUL decodes, never treated as pad/terminator


def test_generate_continuation_only_decode_skips_specials_and_prompt():
    from bytes import BOS_ID as _BOS, EOS_ID as _EOS
    from bytes import SPECIAL_IDS, decode_text
    _SEP = SPECIAL_IDS["<sep>"]
    m = _stub_model([65, _SEP, _EOS])
    res = generate(m, [_BOS, 104, 105], max_new=8, temperature=0.0)
    assert res["ids"][:3] == [_BOS, 104, 105]  # prompt kept, no auto-BOS added
    assert res["ids"][3:] == [65, _SEP, _EOS]  # EOS terminates, SEP kept
    assert res["text"] == decode_text(res["ids"][3:])  # continuation-only
    assert res["text"] == "A"  # SEP/EOS contribute no text
    # No-BOS prompt stays BOS-free: generate never auto-prepends BOS.
    m2 = _stub_model([66])
    res2 = generate(m2, [104, 105], max_new=1, temperature=0.0)
    assert res2["ids"] == [104, 105, 66]
