import pytest

import evaluate


def test_evaluate_rejects_too_small_synthetic_corpus_before_checkpoint_io():
    with pytest.raises(SystemExit) as exc:
        evaluate.main([
            "--preset", "moe-tiny",
            "--seq-len", "32",
            "--synth-bytes", "10",
            "--checkpoint", "/path/that/does/not/exist",
        ])
    assert exc.value.code == 2
