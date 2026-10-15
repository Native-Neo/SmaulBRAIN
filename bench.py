#!/usr/bin/env python3
"""SmaulBRAIN V2 benchmarks: real measurements on this hardware.

Records hardware, CPU features, preset, batch/seq, dtype, quant/paging modes.
Usage: python bench.py [--preset moe-tiny] [--seq-len 64] [--steps 5]
"""
import argparse
import json
import os
import sys
import time

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
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))
    spec = _cfg.PRESETS[preset]
    if spec.name in ("dense-full", "moe-full"):
        print("note: full presets are memory-heavy; running forward-only micro-bench",
              file=sys.stderr)
    import numpy as _np
    init_t0 = time.perf_counter()
    rng = _np.random.default_rng(args.seed)
    params = _model.build_params(spec, rng)
    m = _model.Model(spec, params, forward_quant=args.forward_quant,
                     backward_mode=args.backward_mode)
    model_init_s = time.perf_counter() - init_t0
    toks = list(_np.random.default_rng(1).integers(0, 256, size=args.seq_len))
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
    t0 = time.perf_counter()
    m.forward(toks)
    res["warm_forward_s"] = time.perf_counter() - t0
    res["packed_cache_builds"] = m.packed.builds
    res["packed_cache_hits"] = m.packed.reuse_hits
    res["packed_cache_bytes"] = int(sum(v["packed"].nbytes for v in m.packed.d.values()))
    if spec.name.endswith("tiny"):
        t0 = time.perf_counter()
        loss, grads, _ = m.loss_and_backward(toks)
        res["backward_s"] = time.perf_counter() - t0
        res["loss"] = loss
        res["rss_after_bwd_mb"] = rss_mb()
        from smaulbrain import smaulopt as _opt
        opt = _opt.SmaulOPT()
        opt.register(params)
        t0 = time.perf_counter()
        opt.step(params, grads)
        res["opt_step_s"] = time.perf_counter() - t0
    eng = _inf.InferenceEngine(m, seed=0, temperature=0.0)
    t0 = time.perf_counter()
    eng.prefill(toks)
    t1 = time.perf_counter()
    n = 16 if spec.name.endswith("tiny") else 2
    eng.generate([], max_new_tokens=n)
    t2 = time.perf_counter()
    res["prefill_s"] = t1 - t0
    res["decode_tokens_per_s"] = n / max(t2 - t1, 1e-9)
    res["recurrent_state_bytes"] = spec.n_layers * spec.d_model * 4
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
