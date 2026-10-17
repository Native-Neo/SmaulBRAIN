import train


def test_resume_infers_saved_preset_and_reuses_checkpoint_tensors(tmp_path, capsys, monkeypatch):
    from smaulbrain import model as M

    checkpoint = str(tmp_path / "resume-with-saved-config")
    initial = [
        "--preset", "dense-tiny",
        "--seq-len", "16",
        "--synth-bytes", "128",
        "--seed", "731",
        "--lr", "0.0007",
        "--growth-every-steps", "0",
        "--growth-loss-nats", "none",
        "--checkpoint", checkpoint,
        "--ckpt-every", "0",
        "--max-steps", "1",
    ]
    assert train.main(initial) == 0
    capsys.readouterr()

    def forbidden_reinitialization(*args, **kwargs):
        raise AssertionError("resume must not allocate a newly initialized model")

    monkeypatch.setattr(M, "build_params", forbidden_reinitialization)
    assert train.main([
        "--checkpoint", checkpoint,
        "--ckpt-every", "0",
        "--steps", "1",
    ]) == 0
    output = capsys.readouterr().out

    assert "auto-resumed dense-tiny at global_step=1" in output
    assert '"global_step": 2' in output
    assert '"steps_this_run": 1' in output
