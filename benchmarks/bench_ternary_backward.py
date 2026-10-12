"""Compare the native zero-skipping dX path with dense PyTorch GEMM.

Run on the target CPU; this script reports measurements and makes no
assumption that the native path is faster.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import native


def median_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=3328)
    parser.add_argument("--density", type=float, default=0.35,
                        help="Fraction of nonzero ternary weights (0 < density <= 1).")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.rows <= 0 or args.in_features <= 0 or args.out_features <= 0:
        parser.error("matrix dimensions must be positive")
    if not 0.0 < args.density <= 1.0:
        parser.error("--density must be in (0, 1]")
    if args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("--warmup must be nonnegative; --repeats and --threads must be positive")

    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    grad_out = torch.randn(args.rows, args.out_features, dtype=torch.float32)
    mask = torch.rand(args.out_features, args.in_features) < args.density
    signs = torch.where(
        torch.rand(args.out_features, args.in_features) < 0.5, -1.0, 1.0
    )
    row_scales = torch.rand(args.out_features, 1) + 0.25
    weight = (mask * signs * row_scales).contiguous()
    grad_out = grad_out.contiguous()

    native_result = native.call_ternary_backward_input(grad_out, weight)
    if native_result is None:
        raise RuntimeError(
            "native kernel unavailable; inspect native.status() and ensure g++ is installed"
        )
    dense_result = grad_out @ weight
    if not torch.allclose(native_result, dense_result, atol=1e-4, rtol=1e-4):
        raise RuntimeError("native output differs from dense PyTorch GEMM")

    native_ms = median_ms(
        lambda: native.call_ternary_backward_input(grad_out, weight),
        args.warmup, args.repeats,
    )
    dense_ms = median_ms(lambda: grad_out @ weight, args.warmup, args.repeats)
    print(f"shape: dY={tuple(grad_out.shape)}, W={tuple(weight.shape)}")
    print(f"nonzero density: {args.density:.3f}; torch threads: {args.threads}")
    print(f"native zero-skipping: {native_ms:.3f} ms")
    print(f"dense PyTorch GEMM:   {dense_ms:.3f} ms")
    print(f"speedup (dense/native): {dense_ms / native_ms:.3f}x")
    print(f"native dispatch count: {native.COUNTERS['ternary_backward_native']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
