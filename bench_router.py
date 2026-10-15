#!/usr/bin/env python3
"""Microbenchmark optimized MoE routing against the previous stable-sort path."""
import argparse
import json
import time

import numpy as np

from smaulbrain.moe import Router


def stable_reference(router, X, top_k, capacity_factor=1.25, disabled=()):
    """Previous route algorithm retained here only as a benchmark reference."""
    scores = X @ router.W
    if disabled:
        scores = scores.copy()
        scores[:, list(disabled)] = -np.inf
    T, E = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")
    top_idx = order[:, :top_k]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    ex = np.exp(top_scores - top_scores.max(axis=1, keepdims=True))
    weights = ex / ex.sum(axis=1, keepdims=True)
    cap = max(int(np.ceil(capacity_factor * T * top_k / E)), 1)
    load = np.zeros(E, dtype=np.int64)
    keep = np.ones_like(weights, dtype=bool)
    for t in range(T):
        for j in range(top_k):
            eid = int(top_idx[t, j])
            if load[eid] >= cap:
                keep[t, j] = False
            else:
                load[eid] += 1
    weights = np.where(keep, weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return top_idx, weights, load, keep


def median_seconds(fn, repeat):
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), samples, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if min(args.tokens, args.experts, args.d_model, args.top_k, args.repeat) < 1:
        ap.error("tokens, experts, d-model, top-k and repeat must all be positive")
    if args.top_k > args.experts:
        ap.error("--top-k cannot exceed --experts")

    rng = np.random.default_rng(args.seed)
    router = Router(args.d_model, args.experts, rng)
    X = rng.standard_normal((args.tokens, args.d_model)).astype(np.float32)

    ref_s, ref_samples, ref = median_seconds(
        lambda: stable_reference(router, X, args.top_k), args.repeat)
    opt_s, opt_samples, opt = median_seconds(
        lambda: router.route(X, args.top_k), args.repeat)
    ids, weights, info, cache = opt
    _, ref_weights, ref_load, ref_keep = ref
    if not (np.array_equal(ids, ref[0])
            and np.allclose(weights, ref_weights, atol=1e-6, rtol=1e-6)
            and np.array_equal(info["load"], ref_load)
            and np.array_equal(cache[4], ref_keep)):
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
