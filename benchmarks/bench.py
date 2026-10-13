"""Benchmark harness: throughput, memory, paging, routing, depth, precision.

Measures (never claims without measuring):
  tokens/s (= bytes/s: the vocabulary is bytes) for training and inference,
  peak RSS, checkpoint disk usage, expert loads/evictions per paging mode,
  routing distribution, active experts, recurrent depth, FP8 vs BF16
  expert-compute timing.

Usage: python benchmarks/bench.py [--mode D2R|R2VR|D2VR] [--steps N]
"""

import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import resource

    def rss_mb():
        """Return the process peak resident-set size in MiB."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:  # Windows: resource is Unix-only
    def rss_mb():
        """Return the process peak resident-set size in MiB."""
        return 0.0

import torch
from config import SmaulBrainConfig
from infer import generate
from kernels import time_fn
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import train_step


def disk_mb(path):
    """Measure the total size of files written beneath a directory."""
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            total += os.path.getsize(os.path.join(dp, f))
    return total / 1e6


def main():
    """Validate benchmark options, run measurements, and print a JSON report."""
    ap = argparse.ArgumentParser(description="Measure SmaulBRAIN on the current host.")
    ap.add_argument("--mode", default="D2R", choices=["D2R", "R2VR", "D2VR"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--expert-hidden", type=int, default=128)
    ap.add_argument("--context-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-params", type=int, default=5_000_000,
                    help="Refuse larger models before allocation (default: 5M); 0 disables the guard")
    ap.add_argument("--allow-large", action="store_true",
                    help="Explicitly bypass --max-params; this does not guarantee sufficient RAM")
    args = ap.parse_args()
    for name in ("steps", "batch_size", "d_model", "experts", "top_k",
                 "expert_hidden", "context_length", "min_depth", "max_depth",
                 "threads"):
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_params < 0:
        ap.error("--max-params must be >= 0")
    if args.min_depth > args.max_depth:
        ap.error("--min-depth must be <= --max-depth")
    if args.top_k > args.experts:
        ap.error("--top-k must be <= --experts")
    torch.manual_seed(0)
    try:
        cfg = SmaulBrainConfig(
            d_model=args.d_model, n_heads=4 if args.d_model % 4 == 0 else 1,
            num_experts=args.experts, top_k=args.top_k,
            expert_hidden=args.expert_hidden, min_depth=args.min_depth,
            max_depth=args.max_depth, context_length=args.context_length,
            threads=args.threads, paging_method=args.mode,
        )
    except (AssertionError, ValueError, TypeError) as exc:
        ap.error(f"invalid benchmark configuration: {exc}")
    # Validate first so invalid thread counts/configurations produce a useful
    # argparse error rather than a low-level PyTorch traceback.
    torch.set_num_threads(args.threads)
    estimate = cfg.total_params()
    if args.max_params and estimate > args.max_params and not args.allow_large:
        ap.error(
            f"estimated {estimate:,} parameters exceeds --max-params={args.max_params:,}; "
            "reduce model dimensions or pass --allow-large only if you have checked available RAM"
        )
    print(
        f"Benchmark config: params={estimate:,}, d_model={cfg.d_model}, "
        f"experts={cfg.num_experts}, top_k={cfg.top_k}, expert_hidden={cfg.expert_hidden}, "
        f"context={cfg.context_length}, batch={args.batch_size}, depth={cfg.min_depth}-{cfg.max_depth}",
        file=sys.stderr,
    )
    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    T, B = cfg.context_length, args.batch_size
    ids = torch.randint(0, 256, (B, T + 1))
    x, y = ids[:, :T], ids[:, 1:]

    try:
        # Training throughput.
        t0, tok = time.perf_counter(), 0
        depths = []
        for s in range(args.steps):
            st = train_step(model, opt, cfg, x, y, s)
            tok += B * T
            depths.append(round(st["mean_depth"], 2))
        dt = time.perf_counter() - t0
        train_tps = tok / max(dt, 1e-12)

        # Inference throughput.
        gen_tokens = min(32, max(1, T - 1))
        t0 = time.perf_counter()
        generate(model, [104, 105], max_new=gen_tokens, temperature=0.0, context=T)
        infer_dt = time.perf_counter() - t0
        infer_tps = gen_tokens / max(infer_dt, 1e-12)

        # FP32 vs BF16 expert compute timing (measurement only; not a quality claim).
        from experts import swiglu_forward
        eid = model.pool.order[0]
        wfp32 = {k: v.float() for k, v in model.pool.experts[eid].dequantize(torch.float32).items()}
        wbf16 = {k: v.to(torch.bfloat16) for k, v in wfp32.items()}
        xx = torch.randn(256, cfg.d_model)
        fp32_t = time_fn(lambda: swiglu_forward(xx, wfp32), repeat=10)["median_ms"]
        bf16_t = time_fn(lambda: swiglu_forward(xx.to(torch.bfloat16), wbf16), repeat=10)["median_ms"]

        share = model.router.usage_share().tolist()
        import tempfile
        from storage import save_model
        with tempfile.TemporaryDirectory(prefix="bench_ckpt_") as ckpt_dir:
            save_model(ckpt_dir, model, opt, args.steps)
            report = {
                "paging_mode": args.mode,
                "config": {
                    "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                    "experts": cfg.num_experts, "top_k": cfg.top_k,
                    "expert_hidden": cfg.expert_hidden, "context_length": T,
                    "batch_size": B, "min_depth": cfg.min_depth,
                    "max_depth": cfg.max_depth, "threads": args.threads,
                    "steps": args.steps,
                },
                "train_tokens_per_s": round(train_tps, 2),
                "infer_tokens_per_s": round(infer_tps, 2),
                "peak_rss_mb": round(rss_mb(), 1),
                "checkpoint_disk_mb": round(disk_mb(ckpt_dir), 3),
                "param_counts": model.param_counts(),
                "paging": model.pager.stats.to_dict(),
                "routing_share": [round(v, 4) for v in share],
                "mean_depth_per_step": depths,
                "expert_fp32_ms": round(fp32_t, 3),
                "expert_bf16_ms": round(bf16_t, 3),
            }
        print(json.dumps(report, indent=2))
    finally:
        model.pager.close()


if __name__ == "__main__":
    main()
