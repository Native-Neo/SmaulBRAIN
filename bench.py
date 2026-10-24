#!/usr/bin/env python3
"""SmaulBRAIN v2.1.1 benchmarks: real measurements on this hardware.

Records hardware, CPU features, preset, batch/seq, dtype, quant/paging modes.
Usage: python bench.py [--preset moe-tiny] [--seq-len 64] [--steps 5]
"""
import argparse
import json
import os
import sys
import time
import statistics

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import cpu as _cpu
from smaulbrain import data as _data
from smaulbrain import inference as _inf
from smaulbrain import model as _model
from smaulbrain import quant as _quant


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return -1.0
    return -1.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    _cli.add_common_args(ap)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=5,
                    help="number of warm-forward and decode repetitions (default: 5)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="tokens to generate per decode repetition (default: preset-specific smoke count)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        ap.error("--max-new-tokens must be at least 1")
    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))
    spec = _cfg.PRESETS[preset]
    if spec.name in ("dense-full", "moe-full"):
        print("note: full presets are memory-heavy; running forward-only micro-bench",
              file=sys.stderr)
    init_t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    params = _model.build_params(spec, rng, device=args.device)
    m = _model.Model(spec, params, forward_quant=args.forward_quant,
                     backward_mode=args.backward_mode, device=args.device)
    model_init_s = time.perf_counter() - init_t0
    toks = np.random.default_rng(1).integers(0, 256, (args.seq_len,)).tolist()
    res = {"hw": {"cpu": _cpu.detect(), "dispatch": _cpu.dispatch_name()},
           "preset": preset, "params": m.report()["total_params"],
           "seq_len": args.seq_len, "forward_quant": args.forward_quant,
           "backward_mode": args.backward_mode, "paging": args.paging,
           "model_init_s": model_init_s,
           "ternary_backend": _quant.ternary_backend_name() if args.forward_quant == "ternary" else None,
           "rss_before_mb": rss_mb()}
    t0 = time.perf_counter()
    logits, _ = m.forward(toks)
    res["forward_s"] = time.perf_counter() - t0
    res["rss_after_fwd_mb"] = rss_mb()
    warm_times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        m.forward(toks)
        warm_times.append(time.perf_counter() - t0)
    res["warm_forward_s"] = float(statistics.median(warm_times))
    res["warm_forward_samples_s"] = warm_times
    res["packed_cache_builds"] = m.packed.builds
    res["packed_cache_hits"] = m.packed.reuse_hits
    res["packed_cache_bytes"] = int(sum(
        entry["packed"].nbytes if hasattr(entry["packed"], "nbytes")
        else np.asarray(entry["packed"]).nbytes
        for entry in m.packed.d.values()
    ))

    # Measure inference before the optimizer mutates parameters, so the
    # quantized cache always matches the weights being benchmarked.
    prefill_times, decode_rates = [], []
    n = args.max_new_tokens if args.max_new_tokens is not None else (
        16 if spec.name.endswith("tiny") else 2)
    res["max_new_tokens"] = n
    for _ in range(args.repeat):
        eng = _inf.InferenceEngine(m, seed=0, temperature=0.0)
        t0 = time.perf_counter()
        eng.prefill(toks)
        t1 = time.perf_counter()
        eng.generate([], max_new_tokens=n)
        t2 = time.perf_counter()
        prefill_times.append(t1 - t0)
        decode_rates.append(n / max(t2 - t1, 1e-9))
    res["prefill_s"] = float(statistics.median(prefill_times))
    res["prefill_samples_s"] = prefill_times
    res["decode_tokens_per_s"] = float(statistics.median(decode_rates))
    res["decode_tokens_per_s_samples"] = decode_rates
    res["recurrent_state_bytes"] = spec.n_layers * spec.d_model * 4

    if spec.name.endswith("tiny"):
        t0 = time.perf_counter()
        loss, grads, _ = m.loss_and_backward(toks)
        res["backward_s"] = time.perf_counter() - t0
        res["loss"] = loss
        res["rss_after_bwd_mb"] = rss_mb()
        from smaulbrain import smaulopt as _opt
        opt = _opt.SmaulOPT()
        opt.register(m.p)
        t0 = time.perf_counter()
        opt.step(m.p, grads)
        res["opt_step_s"] = time.perf_counter() - t0
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
