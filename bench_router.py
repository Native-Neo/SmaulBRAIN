#!/usr/bin/env python3
"""Benchmark NumPy MoE routing against a simple stable-sort reference."""
import argparse
import json
import statistics
import time

import numpy as np

from smaulbrain.moe import Router


def stable_reference(router, X, top_k, capacity_factor=1.25, disabled=()):
    scores = X @ router.W
    if disabled:
        scores = scores.copy()
        scores[:, list(disabled)] = -np.inf
    T, E = scores.shape
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    values = np.take_along_axis(scores, idx, axis=1)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / np.sum(exp, axis=1, keepdims=True)
    cap = max(int((capacity_factor * T * top_k / E) + 0.999999), 1)
    load = np.zeros(E, dtype=np.int64)
    keep = np.ones_like(idx, dtype=bool)
    for t in range(T):
        for j in range(top_k):
            e = int(idx[t, j])
            if load[e] >= cap:
                keep[t, j] = False
            else:
                load[e] += 1
    weights = weights * keep
    weights = weights / np.maximum(weights.sum(1, keepdims=True), 1e-12)
    return idx, weights, load, keep


def median_seconds(fn, repeat):
    samples = []
    result = None
    for _ in range(repeat):
        t = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t)
    return float(statistics.median(samples)), samples, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if min(args.tokens, args.experts, args.d_model, args.top_k, args.repeat) < 1:
        ap.error("all sizes must be positive")
    if args.top_k > args.experts:
        ap.error("--top-k cannot exceed --experts")
    rng = np.random.default_rng(args.seed)
    router = Router(args.d_model, args.experts, rng=rng, device=args.device)
    X = rng.standard_normal((args.tokens, args.d_model)).astype(np.float32)
    ref_s, ref_samples, ref = median_seconds(
        lambda: stable_reference(router, X, args.top_k), args.repeat
    )
    opt_s, opt_samples, opt = median_seconds(
        lambda: router.route(X, args.top_k), args.repeat
    )
    ids, weights, info, cache = opt
    parity = (
        np.array_equal(ids, ref[0])
        and np.allclose(weights, ref[1], atol=1e-6, rtol=1e-6)
        and np.array_equal(np.asarray(info["load"], dtype=np.int64), ref[2])
        and np.array_equal(cache[4], ref[3])
    )
    if not parity:
        raise SystemExit("optimized route diverged from stable reference")
    print(json.dumps({
        "shape": {"tokens": args.tokens, "experts": args.experts,
                  "d_model": args.d_model, "top_k": args.top_k},
        "repeat": args.repeat,
        "stable_reference_median_s": ref_s,
        "stable_reference_samples_s": ref_samples,
        "optimized_median_s": opt_s,
        "optimized_samples_s": opt_samples,
        "speedup": ref_s / max(opt_s, 1e-12),
        "parity": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
