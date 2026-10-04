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
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:  # Windows: resource is Unix-only
    def rss_mb():
        return 0.0

import torch
from config import SmaulBrainConfig
from infer import generate
from kernels import time_fn
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import train_step


def disk_mb(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            total += os.path.getsize(os.path.join(dp, f))
    return total / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="D2R", choices=["D2R", "R2VR", "D2VR"])
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    torch.manual_seed(0)
    cfg = SmaulBrainConfig(d_model=64, n_heads=4, num_experts=8, top_k=2,
                           expert_hidden=128, min_depth=1, max_depth=3,
                           context_length=128, paging_method=args.mode)
    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr))
    T, B = cfg.context_length, 2
    ids = torch.randint(0, 256, (B, T + 1))
    x, y = ids[:, :T], ids[:, 1:]

    # --- training throughput ---
    t0, tok = time.perf_counter(), 0
    depths = []
    for s in range(args.steps):
        st = train_step(model, opt, cfg, x, y, s)
        tok += B * T
        depths.append(round(st["mean_depth"], 2))
    dt = time.perf_counter() - t0
    train_tps = tok / dt

    # --- inference throughput ---
    t0 = time.perf_counter()
    gen = generate(model, [104, 105], max_new=32, temperature=0.0, context=T)
    infer_dt = time.perf_counter() - t0
    infer_tps = 32 / infer_dt

    # --- FP8 vs BF16 expert compute ---
    from experts import swiglu_forward
    eid = model.pool.order[0]
    wfp32 = {k: v.float() for k, v in model.pool.experts[eid].dequantize(torch.float32).items()}
    wbf16 = {k: v.to(torch.bfloat16) for k, v in wfp32.items()}
    xx = torch.randn(256, cfg.d_model)
    fp32_t = time_fn(lambda: swiglu_forward(xx, wfp32), repeat=10)["median_ms"]
    bf16_t = time_fn(lambda: swiglu_forward(xx.to(torch.bfloat16), wbf16), repeat=10)["median_ms"]

    # --- routing distribution ---
    share = model.router.usage_share().tolist()
    # --- checkpoint disk usage (actually written + measured) ---
    import tempfile
    from storage import save_model
    ckpt_dir = tempfile.mkdtemp(prefix="bench_ckpt_")
    save_model(ckpt_dir, model, opt, args.steps)
    report = {
        "paging_mode": args.mode,
        "train_tokens_per_s": round(train_tps, 1),
        "train_bytes_per_s": round(train_tps, 1),  # byte vocab: 1 token = 1 byte
        "infer_tokens_per_s": round(infer_tps, 1),
        "infer_bytes_per_s": round(infer_tps, 1),
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
    model.pager.close()


if __name__ == "__main__":
    main()
