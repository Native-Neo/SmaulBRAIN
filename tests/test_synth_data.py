"""Synthetic data generator: allowlist, parsing, disjointness (mocked runner)."""

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import synth_data as sd


class _Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _text_event(text):
    return json.dumps({"type": "text", "part": {"type": "text", "text": text}})


def test_normalize_accepts_free_short_and_prefixed():
    assert sd.normalize_model("big-pickle") == "opencode/big-pickle"
    assert sd.normalize_model("opencode/exo-free") == "opencode/exo-free"
    for m in sd.FREE_MODELS:
        assert sd.normalize_model(m) == f"opencode/{m}"


def test_normalize_refuses_paid_unknown_and_jev():
    for bad in ("gpt-5.5", "opencode/gpt-5.5", "claude-sonnet-4-5",
                "jev-1.13-free", "nope", ""):
        with pytest.raises(ValueError):
            sd.normalize_model(bad)


def test_list_models_prints_allowlist_only(capsys):
    assert sd.main(["--list-models"]) == 0
    out = capsys.readouterr().out
    for m in sd.FREE_MODELS:
        assert m in out
    assert "gpt-5" not in out and "jev-1.13-free" not in out


def test_rejects_nonfree_model_exit_2():
    assert sd.main(["--model", "gpt-5.5", "--dry-run"]) == 2


def test_extract_text_collects_assistant_parts_ignores_tools():
    events = "\n".join([
        _text_event("hello "),
        json.dumps({"type": "tool", "part": {"type": "tool", "name": "bash"}}),
        "not json at all",
        _text_event("world"),
        json.dumps({"type": "text", "part": {"type": "text", "text": 42}}),
    ])
    assert sd.extract_text(events) == "hello world"
    assert sd.extract_text("garbage\nmore garbage") == ""


def test_clean_lines_strips_markers_and_non_ascii():
    text = "```\n1. hello syslog line one\n- second line here\n\n  café ünïcode  \n```\n"
    lines = sd.clean_lines(text)
    assert lines and all(32 <= ord(c) < 127 for l in lines for c in l)
    assert not any(l[0].isdigit() and l[1:3] in (". ", ") ") for l in lines)


def _fake_runner(texts):
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        assert cmd[1] == "run" and cmd[2] == "--model"
        body = texts[min(len(calls) - 1, len(texts) - 1)]
        return _Proc(_text_event(body))
    run.calls = calls
    return run


def test_generate_domain_disjoint_and_sized(tmp_path):
    train_pool = "\n".join(f"syslog disk error number {i} on host abc" for i in range(30))
    held_pool = "\n".join(f"recipe stir sugar batch {i} grams flour" for i in range(30))
    run = _fake_runner([train_pool, held_pool])
    train, held = sd.generate_domain("x", ["t1", "t2", "t3", "t4"], "style",
                                     "opencode/big-pickle", 10, 6, 0, 60,
                                     per_call=30, runner=run)
    assert len(train) == 10 and len(held) == 6
    assert not (set(train) & set(held))
    # Model id actually passed to the CLI.
    assert all(c[3] == "opencode/big-pickle" for c in run.calls)


def test_generate_domain_retries_then_raises():
    def boom(cmd, **kw):
        return _Proc("", returncode=1, stderr="auth missing")
    with pytest.raises(RuntimeError):
        sd.generate_domain("x", ["t1", "t2"], "style", "opencode/big-pickle",
                           4, 2, 0, 60, per_call=4, retries=2, runner=boom)


def test_generate_domain_empty_text_retries():
    n = {"i": 0}
    def flaky(cmd, **kw):
        n["i"] += 1
        if n["i"] < 2:
            return _Proc(_text_event("   "))
        return _Proc(_text_event("good syslog line number one here"))
    train, _ = sd.generate_domain("x", ["t1", "t2"], "style",
                                  "opencode/big-pickle", 1, 0, 0, 60,
                                  per_call=4, runner=flaky)
    assert train == ["good syslog line number one here"]


def test_write_domain_files_and_hashes(tmp_path):
    d = str(tmp_path / "o")
    info = sd.write_domain(d, "A", ["a1", "a2"], ["h1"])
    assert info["train"]["lines"] == 2 and info["held"]["lines"] == 1
    assert open(info["train"]["path"]).read() == "a1\na2\n"
    assert len(info["train"]["sha256"]) == 64


def test_dry_run_makes_no_calls():
    def boom(cmd, **kw):
        raise AssertionError("must not call opencode")
    assert sd.main(["--dry-run"]) == 0
