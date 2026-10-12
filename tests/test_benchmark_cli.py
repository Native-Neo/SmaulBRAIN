"""CLI-level regression tests for benchmark safety and validation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "bench.py"


def run_bench(*args: str) -> subprocess.CompletedProcess[str]:
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
    result = run_bench(*args)
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_benchmark_parameter_guard_refuses_before_model_allocation():
    result = run_bench("--max-params", "1")
    assert result.returncode == 2
    assert "exceeds --max-params=1" in result.stderr
    assert "Traceback" not in result.stderr


def test_benchmark_allows_explicit_guard_override():
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
