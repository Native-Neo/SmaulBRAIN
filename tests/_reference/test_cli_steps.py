import json

import pytest
import torch

import train


def test_resume_steps_means_additional_updates(tmp_path, capsys):
    ckpt = str(tmp_path / "run")
    common = ["--preset", "moe-tiny", "--seq-len", "16", "--growth-every-steps", "0",
              "--growth-loss-nats", "none", "--checkpoint", ckpt, "--log-every", "1"]
    assert train.main(common + ["--max-steps", "1"]) == 0
    capsys.readouterr()
    assert train.main(common + ["--resume", "--steps", "2"]) == 0
    output = capsys.readouterr().out
    assert '"steps_this_run": 2' in output
    assert '"last_loss":' in output


@pytest.mark.parametrize("option", ["--steps", "--max-steps"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_updates_are_rejected(option, value, capsys, monkeypatch):
    def unexpected_training(argv):
        pytest.fail("invalid step count reached ncm_train")

    monkeypatch.setattr(train, "_ncm_train", unexpected_training)
    with pytest.raises(SystemExit) as exc:
        train.main([option, value])
    assert exc.value.code == 2
    assert f"{option} must be >= 1" in capsys.readouterr().err


@pytest.fixture
def small_run(tmp_path):
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes(b"small repeatable training corpus\n" * 8)
    checkpoint = tmp_path / "run"
    common = [
        "--preset", "dense-tiny", "--d-model", "8", "--layers", "1",
        "--seq-len", "8", "--levels", "2", "--codebook-size", "8",
        "--batch-size", "1", "--device", "cpu",
        "--corpus", str(corpus), "--checkpoint", str(checkpoint),
    ]
    return checkpoint, common


def _summary(capsys):
    output = capsys.readouterr().out
    return json.loads(output[output.rfind('{\n  "done": true,'):])


def _saved_step(checkpoint, stage):
    return torch.load(checkpoint / f"{stage}-training.pt", weights_only=True)["step"]


def test_max_steps_targets_saved_stage_steps_and_steps_adds_updates(small_run, capsys):
    checkpoint, common = small_run
    assert train.main(common + ["--stage", "tokenizer", "--max-steps", "1"]) == 0
    assert _summary(capsys)["tokenizer_steps_this_run"] == 1

    # Only the tokenizer has a saved step; each stage must reach the same target.
    assert train.main(common + ["--resume", "--max-steps", "3"]) == 0
    summary = _summary(capsys)
    assert summary["tokenizer_steps_this_run"] == 2
    assert summary["generator_steps_this_run"] == 3
    assert _saved_step(checkpoint, "tokenizer") == 3
    assert _saved_step(checkpoint, "generator") == 3
    frozen = {
        name: (checkpoint / name).read_bytes()
        for name in ("tokenizer.pt", "tokenizer-training.pt")
    }

    assert train.main(common + ["--resume", "--max-steps", "5"]) == 0
    summary = _summary(capsys)
    assert summary["tokenizer_steps_this_run"] == 0
    assert summary["generator_steps_this_run"] == 2
    assert _saved_step(checkpoint, "generator") == 5
    generator_before = (checkpoint / "generator-training.pt").read_bytes()

    for target in ("5", "4"):
        assert train.main(common + ["--resume", "--max-steps", target]) == 0
        summary = _summary(capsys)
        assert summary["tokenizer_steps_this_run"] == 0
        assert summary["generator_steps_this_run"] == 0
        assert summary["steps_this_run"] == 0
        assert (checkpoint / "generator-training.pt").read_bytes() == generator_before

    # An explicit additional-update count takes precedence, even over an invalid target.
    assert train.main(common + ["--resume", "--steps", "2", "--max-steps", "0"]) == 0
    summary = _summary(capsys)
    assert summary["generator_steps_this_run"] == 2
    assert _saved_step(checkpoint, "generator") == 7
    for name, contents in frozen.items():
        assert (checkpoint / name).read_bytes() == contents

    # Explicit tokenizer training remains available after generator training.
    assert train.main(common + ["--stage", "tokenizer", "--steps", "1"]) == 0
    assert _summary(capsys)["tokenizer_steps_this_run"] == 1
    assert _saved_step(checkpoint, "tokenizer") == 4


@pytest.mark.parametrize("problem", ["missing", "mismatched"])
def test_both_resume_validates_saved_tokenizer(small_run, capsys, problem):
    checkpoint, common = small_run
    assert train.main(common + ["--steps", "1"]) == 0
    capsys.readouterr()
    tokenizer_training = (checkpoint / "tokenizer-training.pt").read_bytes()
    if problem == "missing":
        (checkpoint / "tokenizer.pt").unlink()
        message = "missing tokenizer checkpoint"
    else:
        common += ["--levels", "3"]
        message = "CLI model settings do not match the saved tokenizer configuration"
    with pytest.raises(SystemExit) as exc:
        train.main(common + ["--resume", "--steps", "1"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert (checkpoint / "tokenizer-training.pt").read_bytes() == tokenizer_training
