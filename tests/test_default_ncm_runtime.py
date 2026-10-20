import json

import pytest
import torch

import ncm_bench
from ncm import NCMConfig, NCMGenerator, ResidualVQTokenizer
from infer import main as infer_main
from train import main as train_main


def test_default_cli_trains_and_infers_with_packed_ternary(tmp_path, capsys):
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes((b"small byte corpus with repeated words\n" * 16))
    checkpoint = tmp_path / "ncm-checkpoint"

    assert train_main([
        "--preset", "dense-tiny",
        "--d-model", "8",
        "--layers", "1",
        "--seq-len", "16",
        "--levels", "2",
        "--codebook-size", "8",
        "--batch-size", "1",
        "--steps", "2",
        "--save-every", "2",
        "--log-every", "1",
        "--corpus", str(corpus),
        "--checkpoint", str(checkpoint),
        "--forward-quant", "ternary",
        "--backward-mode", "ternary",
    ]) == 0
    training_output = capsys.readouterr().out
    summary_start = training_output.rfind('{\n  "done": true,')
    assert summary_start >= 0
    summary = json.loads(training_output[summary_start:])
    assert summary["optimizer"] == "SmaulOPT"
    assert summary["tokenizer_steps_this_run"] == 2
    assert summary["generator_steps_this_run"] == 2
    assert "tokenizer step=1 " in training_output
    assert "tokenizer step=2 " in training_output
    assert "generator step=1 " in training_output
    assert "generator step=2 " in training_output

    assert infer_main([
        "--checkpoint", str(checkpoint),
        "--prompt", "Hi",
        "--max-new-tokens", "4",
    ]) == 0
    inference_output = capsys.readouterr().out
    inference = json.loads(inference_output)
    assert inference["preset"] == "dense-tiny"
    assert inference["forward_quant"] == "ternary"
    assert inference["backward_mode"] == "ternary"
    assert inference["generated_bytes"] == 4


@pytest.fixture
def inference_checkpoint(tmp_path):
    config = NCMConfig(
        d_model=8, n_layers=1, levels=2, codebook_size=8,
        max_seq_len=8, n_experts=2, top_k=1,
    )
    tokenizer = ResidualVQTokenizer(config)
    generator = NCMGenerator(config, tokenizer)
    for name, model in (("tokenizer", tokenizer), ("generator", generator)):
        torch.save({"config": config.__dict__, "model": model.state_dict()}, tmp_path / f"{name}.pt")
    return tmp_path


@pytest.mark.parametrize("seed", [None, 0, 271])
def test_inference_applies_only_explicit_seed(inference_checkpoint, monkeypatch, capsys, seed):
    manual_seed = torch.manual_seed
    seeds = []

    def record_seed(value):
        seeds.append(value)
        return manual_seed(value)

    monkeypatch.setattr(torch, "manual_seed", record_seed)
    args = [
        "--checkpoint", str(inference_checkpoint), "--prompt", "Hi",
        "--max-new-tokens", "8", "--temperature", "1.0",
    ]
    if seed is not None:
        args += ["--seed", str(seed)]
    outputs = []
    for _ in range(2):
        assert infer_main(args) == 0
        outputs.append(json.loads(capsys.readouterr().out)["text_utf8"])
    assert seeds == ([] if seed is None else [seed, seed])
    if seed is not None:
        assert outputs[0] == outputs[1]


@pytest.mark.parametrize("entrypoint", ["infer", "bench"])
def test_inference_loads_only_weights(inference_checkpoint, monkeypatch, capsys, entrypoint):
    load = torch.load
    loaded = []

    def load_weights(path, **kwargs):
        assert kwargs["weights_only"] is True
        assert torch.device(kwargs["map_location"]) == torch.device("cpu")
        loaded.append(path)
        return load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load_weights)
    if entrypoint == "infer":
        result = infer_main([
            "--checkpoint", str(inference_checkpoint), "--prompt", "Hi",
            "--max-new-tokens", "4", "--device", "cpu",
        ])
    else:
        result = ncm_bench.main([
            "--checkpoint-dir", str(inference_checkpoint), "--prompt", "Hi",
            "--new-bytes", "4", "--runs", "1", "--warmup", "0", "--device", "cpu",
        ])
    assert result == 0
    assert loaded == [inference_checkpoint / name for name in ("tokenizer.pt", "generator.pt")]
    capsys.readouterr()
