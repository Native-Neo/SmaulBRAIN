import json

import numpy as np
import pytest

import ncm
from ncm_checkpoint import load_checkpoint
from smaulbrain import npgrad as ng
from smaulbrain import safetensors_np


def _config(kind="moe"):
    return ncm.NCMConfig(
        vocab_size=256, d_model=8, codebook_size=8, levels=2,
        n_layers=1, n_heads=2, max_seq_len=8, dropout=0.0,
        model_kind=kind, n_experts=2 if kind == "moe" else 0,
        top_k=1 if kind == "moe" else 0,
    )


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_ncm_forward_backward_and_generation(kind):
    cfg = _config(kind)
    ng.manual_seed(3)
    tok = ncm.ResidualVQTokenizer(cfg).train()
    data = ng.Tensor(np.random.default_rng(4).integers(0, 256, (2, 8), dtype=np.int64))
    logits, loss, codes = tok(data)
    loss.backward()
    assert np.isfinite(loss.item())
    assert len(logits) == cfg.levels and len(codes) == cfg.levels
    assert all(p.grad is not None and np.isfinite(p.grad).all() for p in tok.parameters())

    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    gen = ncm.NCMGenerator(cfg, tok).train()
    gen.eval()
    with ng.no_grad():
        codes = tok.encode(data)
        level_loss = ncm.ncm_loss(gen, codes)
    assert np.isfinite(level_loss.item())

    ng.manual_seed(9)
    out1 = gen.generate(6, prefix_tokens=ng.Tensor(np.asarray([[65, 66]], dtype=np.int64)))
    ng.manual_seed(9)
    out2 = gen.generate(6, prefix_tokens=ng.Tensor(np.asarray([[65, 66]], dtype=np.int64)))
    assert out1.shape == (1, 6)
    assert np.array_equal(out1.data, out2.data)


def test_checkpoint_roundtrip_is_safetensors(tmp_path):
    from ncm_train import SmaulTrainingOptimizer, save_checkpoint, load_stage_checkpoint

    cfg = _config("moe")
    ng.manual_seed(1)
    model = ncm.ResidualVQTokenizer(cfg)
    opt = SmaulTrainingOptimizer(model, lr=1e-3)
    path = tmp_path / "tokenizer-training.safetensors"
    save_checkpoint(path, model, opt, cfg, 7)
    tensors, metadata = safetensors_np.load_file(path)
    assert metadata["kind"] == "ncm-train"
    assert any(name.startswith("model::") for name in tensors)

    blob = load_checkpoint(path)
    assert blob["step"] == 7 and blob["optimizer"] is not None
    assert ncm.NCMConfig(**blob["config"]) == cfg
    restored = ncm.ResidualVQTokenizer(cfg)
    restored.load_state_dict(blob["model"])
    for (name, a), (name2, b) in zip(model.state_dict().items(),
                                     restored.state_dict().items()):
        assert name == name2
        assert np.array_equal(a, b)

    opt2 = SmaulTrainingOptimizer(restored, lr=1e-3)
    step = load_stage_checkpoint(path, restored, opt2)
    assert step == 7
    assert opt2.optimizer.n_updates == opt.optimizer.n_updates


def test_train_and_infer_cli_smoke(tmp_path, capsys):
    import infer
    import ncm_train

    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes((b"the quick brown fox jumps over the lazy dog. " * 8))
    checkpoint = tmp_path / "run"
    args = [
        "--data", str(corpus), "--output", str(checkpoint),
        "--stage", "both", "--steps", "1", "--batch-size", "1",
        "--seq-len", "8", "--levels", "2", "--codebook-size", "8",
        "--d-model", "8", "--layers", "1", "--experts", "2", "--top-k", "1",
        "--save-every", "1", "--log-every", "1",
    ]
    assert ncm_train.main(args) == 0
    assert (checkpoint / "tokenizer.safetensors").is_file()
    assert (checkpoint / "generator.safetensors").is_file()
    capsys.readouterr()

    assert infer.main([
        "--checkpoint", str(checkpoint), "--prompt", "Hi",
        "--max-new-tokens", "4", "--temperature", "0",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["generated_bytes"] == 4
    assert report["preset"] == "custom"
