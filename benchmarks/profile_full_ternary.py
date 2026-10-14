"""Profile one full-size CPU training step with ternary QAT enabled.

The input is deterministic synthetic byte data, so this benchmark can run in
CI without private/local corpus files. It matches the --full model dimensions
and 2 x 1024 training tokens, then reports the slowest PyTorch CPU operators.
Profiler overhead affects wall time; use the operator ranking to find hotspots,
not as a throughput claim.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import SmaulBrainConfig
from model import SmaulBrainModel
from native import COUNTERS, LAST_FALLBACK
from smaulopt import SmaulOpt, SmaulOptHParams
from train import train_step


def peak_rss_mib() -> float | None:
    """Return process peak RSS in MiB on Linux/macOS."""
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        return value / (1024.0 if sys.platform.startswith("linux") else 1024.0**2)
    except (AttributeError, OSError):
        return None


def cpu_model() -> str:
    """Read the first available CPU model label without adding dependencies."""
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[-1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> int:
    threads = int(os.environ.get("SMAUL_PROFILE_THREADS", "2"))
    torch.set_num_threads(threads)
    torch.manual_seed(0)

    # Matches cli.py FULL_DEFAULTS plus the exact precision flags in the slow
    # command. Synthetic ids avoid depending on an untracked local corpus.
    cfg = SmaulBrainConfig(
        d_model=512,
        n_heads=4,
        num_experts=64,
        top_k=8,
        active_experts=8,
        max_experts=128,
        min_experts=64,
        expert_hidden=3328,
        min_depth=1,
        max_depth=3,
        halting_threshold=0.9,
        context_length=1024,
        ram_cache=32,
        vram_cache=16,
        dtype="bf16",
        state_dtype="bf16",
        threads=threads,
        ternary_qat=True,
    )

    print(
        f"Profile host: {cpu_model()} | Python {platform.python_version()} | "
        f"PyTorch {torch.__version__} | threads={torch.get_num_threads()}",
        flush=True,
    )
    print(
        "Profile shape: batch=2, context=1024, experts=64, top_k=8, "
        "hidden=3328, depth=1..3, dtype=bf16, ternary_qat=true, "
        flush=True,
    )

    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    # These model-side timers isolate routed forward time from the profiler's
    # operator table. The latter also covers backward and optimizer work.
    model._profile_enabled = True
    model._profile_times = {"router_s": 0.0, "expert_s": 0.0, "balance_s": 0.0}
    ids = torch.randint(0, 256, (2, 1025), dtype=torch.long)
    x, y = ids[:, :-1], ids[:, 1:]

    started = time.perf_counter()
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            stats = train_step(model, opt, cfg, x, y, 0, profile=True)
        wall_s = time.perf_counter() - started

        # Aggregate across shapes for compact, stable CI logs.
        averages = prof.key_averages()
        averages = sorted(
            averages,
            key=lambda item: item.self_cpu_time_total,
            reverse=True,
        )
        top_ops = [
            {
                "operator": item.key,
                "calls": item.count,
                "self_cpu_ms": round(item.self_cpu_time_total / 1000.0, 3),
                "total_cpu_ms": round(item.cpu_time_total / 1000.0, 3),
            }
            for item in averages[:40]
        ]
        native = {
            "ternary_linear_forward_native": int(COUNTERS.get("ternary_linear_forward_native", 0)),
            "ternary_linear_forward_fallback": int(COUNTERS.get("ternary_linear_forward_fallback", 0)),
            "last_packed_forward_fallback": LAST_FALLBACK.get("ternary_linear_forward"),
        }
        report = {
            "host": {
                "cpu": cpu_model(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "threads": torch.get_num_threads(),
            },
            "config": {
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "experts": cfg.num_experts,
                "top_k": cfg.top_k,
                "expert_hidden": cfg.expert_hidden,
                "batch_size": 2,
                "context_length": cfg.context_length,
                "min_depth": cfg.min_depth,
                "max_depth": cfg.max_depth,
                "dtype": cfg.dtype,
                "ternary_qat": cfg.ternary_qat,
            },
            "step": {
                "wall_seconds_including_profiler": round(wall_s, 3),
                "peak_rss_mib": round(peak_rss_mib(), 1) if peak_rss_mib() is not None else None,
                "loss": float(stats["loss"]),
                "mean_depth": float(stats.get("mean_depth", 0.0)),
                "model_forward_timers_seconds": {
                    key: round(float(value), 3)
                    for key, value in model._profile_times.items()
                },
                "native_dispatch": native,
            },
            "top_cpu_operators": top_ops,
        }
        output = Path("ternary-profile-report.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("\nTop CPU operators by self time (profiling overhead included):", flush=True)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25), flush=True)
        print("\nProfile summary:", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Report written to {output}", flush=True)
    finally:
        model.pager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
