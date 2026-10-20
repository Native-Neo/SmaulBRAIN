import json

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
        "--steps", "1",
        "--save-every", "1",
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
    assert summary["tokenizer_steps_this_run"] == 1
    assert summary["generator_steps_this_run"] == 1

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
