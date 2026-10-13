"""CLI-level regression tests for benchmark safety and validation."""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "bench.py"


def run_bench(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the benchmark CLI with a timeout and capture its output and exit code."""
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    return subprocess.run(
        [sys.executable, str(BENCH), *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--threads", "0"), "--threads must be >= 1"),
        (("--steps", "0"), "--steps must be >= 1"),
        (("--batch-size", "0"), "--batch-size must be >= 1"),
        (("--min-depth", "3", "--max-depth", "2"), "--min-depth must be <= --max-depth"),
        (("--experts", "2", "--top-k", "3"), "--top-k must be <= --experts"),
    ],
)
def test_benchmark_rejects_invalid_arguments_cleanly(args, message):
    """Invalid benchmark options report an argparse error without a traceback."""
    result = run_bench(*args)
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_benchmark_parameter_guard_refuses_before_model_allocation():
    """The parameter limit rejects an oversized model before allocation."""
    result = run_bench("--max-params", "1")
    assert result.returncode == 2
    assert "exceeds --max-params=1" in result.stderr
    assert "Traceback" not in result.stderr


def test_benchmark_allows_explicit_guard_override():
    """An explicit limit override lets a tiny benchmark finish and report metrics."""
    # The guard is bypassed explicitly; use tiny dimensions so this remains a
    # real end-to-end smoke test rather than a memory-intensive benchmark.
    result = run_bench(
        "--steps", "1", "--d-model", "8", "--experts", "1", "--top-k", "1",
        "--expert-hidden", "8", "--context-length", "2", "--batch-size", "1",
        "--max-params", "1", "--allow-large",
    )
    assert result.returncode == 0, result.stderr
    assert '"train_tokens_per_s"' in result.stdout
    assert '"peak_rss_mb"' in result.stdout


@pytest.mark.parametrize("byte_tokens", [0, 2])
def test_benchmark_counts_generated_tokens_when_eos_stops_early(monkeypatch, capsys, byte_tokens):
    """Measure actual continuation steps, including EOS and excluding the prompt."""
    from types import SimpleNamespace

    from benchmarks import bench
    from bytes import EOS_ID
    import infer

    sampled = iter([104] * byte_tokens + [EOS_ID])
    monkeypatch.setattr(infer, "sample_next", lambda *args, **kwargs: next(sampled))
    times = iter([0.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(bench, "time", SimpleNamespace(perf_counter=lambda: next(times)))
    monkeypatch.setattr(sys, "argv", [
        str(BENCH), "--steps", "1", "--d-model", "8", "--experts", "1",
        "--top-k", "1", "--expert-hidden", "8", "--context-length", "8",
        "--batch-size", "1", "--threads", "1",
    ])

    bench.main()

    report = json.loads(capsys.readouterr().out)
    assert report["infer_tokens_per_s"] == (byte_tokens + 1) / 2.0
