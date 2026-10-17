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


def test_existing_checkpoint_auto_resumes_without_flag(tmp_path, capsys):
    ckpt = str(tmp_path / "auto-run")
    common = ["--preset", "moe-tiny", "--seq-len", "16", "--growth-every-steps", "0",
              "--growth-loss-nats", "none", "--checkpoint", ckpt, "--log-every", "1"]
    assert train.main(common + ["--max-steps", "1"]) == 0
    capsys.readouterr()
    assert train.main(common + ["--max-steps", "2"]) == 0
    output = capsys.readouterr().out
    assert "auto-resumed moe-tiny at global_step=1" in output
    assert '"global_step": 2' in output
