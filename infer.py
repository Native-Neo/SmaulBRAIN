#!/usr/bin/env python3
"""SmaulBRAIN v2.1.1 inference entry point (training-independent path)."""
import argparse
import json
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import inference as _inf
from smaulbrain import model as _model


def _was_provided(argv, option):
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def main(argv=None):
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Run SmaulBRAIN v2.1.1 inference",
                                 allow_abbrev=False)
    _cli.add_common_args(ap)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--stop-eos", action="store_true")
    args = ap.parse_args(raw_argv)

    if args.max_new_tokens < 0:
        ap.error("--max-new-tokens must be >= 0")
    if args.prefill_chunk < 0:
        ap.error("--prefill-chunk must be >= 0")
    if args.temperature < 0 or not math.isfinite(args.temperature):
        ap.error("--temperature must be a finite number >= 0")
    if args.top_k < 0:
        ap.error("--top-k must be >= 0")
    if not 0 < args.top_p <= 1 or not math.isfinite(args.top_p):
        ap.error("--top-p must be finite and in (0, 1]")

    man_path = os.path.join(args.checkpoint, "manifest.json")
    backup_manifest = os.path.join(os.path.normpath(args.checkpoint) + ".bak",
                                  "manifest.json")
    if not os.path.exists(man_path) and not os.path.exists(backup_manifest):
        print(f"error: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 2

    from smaulbrain import checkpoint as _ckpt
    try:
        blob = _ckpt.load_checkpoint(args.checkpoint)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: unable to load checkpoint at {args.checkpoint}: {exc}",
              file=sys.stderr)
        return 2

    manifest = blob["manifest"]
    saved_config = manifest.get("config", {}) or {}

    if (not _was_provided(raw_argv, "--preset")
            and not args.moe and not args.dense):
        args.preset = manifest.get("preset", args.preset)

    for option, attr, key in (
        ("--forward-quant", "forward_quant", "forward_quant"),
        ("--backward-mode", "backward_mode", "backward_mode"),
        ("--paging", "paging", "paging"),
        ("--seed", "seed", "seed"),
        ("--seq-len", "seq_len", "seq_len"),
        ("--expert-capacity-factor", "expert_capacity_factor", "capacity_factor"),
    ):
        saved = saved_config.get(key)
        if saved is not None and not _was_provided(raw_argv, option):
            setattr(args, attr, saved)

    if args.device == "cpu" and args.paging in ("V", "D2V", "R2V"):
        args.paging = "R"

    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))

    if manifest.get("preset") != preset:
        print(f"error: checkpoint preset {manifest.get('preset')} != {preset}",
              file=sys.stderr)
        return 2

    spec = _cfg.PRESETS[preset]

    mtp_depth = int(saved_config.get("mtp_depth", 0))
    mtp_loss_weight = float(saved_config.get("mtp_loss_weight", 0.1))
    capacity_factor = float(args.expert_capacity_factor)

    store = None
    temp_expert_dir = None
    if spec.kind == "moe":
        from smaulbrain import paging as _paging
        n_exp = manifest["n_experts_current"]
        if args.paging in ("R", "V", "R2V"):
            ram = {e: {"Win": blob["params"][f"expert.{e}.Win"],
                       "Wout": blob["params"][f"expert.{e}.Wout"]} for e in range(n_exp)}
            store = _paging.ExpertStore(args.paging, n_exp, ram_weights=ram)
        else:
            temp_expert_dir = tempfile.TemporaryDirectory(prefix="smaul-infer-")
            dd = temp_expert_dir.name
            import torch
            for e in range(n_exp):
                torch.save({"Win": blob["params"][f"expert.{e}.Win"].detach().cpu(),
                            "Wout": blob["params"][f"expert.{e}.Wout"].detach().cpu()},
                           os.path.join(dd, f"expert_{e}.pt"))
            store = _paging.ExpertStore(args.paging, n_exp, disk_dir=dd,
                                       ram_cache_max_experts=args.paging_cache_experts,
                                       staging_max_bytes=args.paging_staging_bytes)

    m = _model.Model(spec, dict(blob["params"]), forward_quant=args.forward_quant,
                     backward_mode=args.backward_mode, paging_mode=args.paging,
                     paging_store=store, capacity_factor=capacity_factor,
                     mtp_depth=mtp_depth, mtp_loss_weight=mtp_loss_weight,
                     device=args.device)
    eng = _inf.InferenceEngine(m, seed=args.seed, temperature=args.temperature,
                               top_k=args.top_k, top_p=args.top_p)
    if args.prompt_file:
        with open(args.prompt_file, "rb") as f:
            prompt = list(f.read())
    else:
        prompt = list(args.prompt.encode("utf-8", errors="surrogateescape"))
    from smaulbrain import vocab as _vocab
    stop = [_vocab.EOS_ID] if args.stop_eos else []
    t0 = time.perf_counter()
    eng.prefill(prompt, chunk=args.prefill_chunk)
    t1 = time.perf_counter()
    out = []
    for tok in eng.generate_stream([], max_new_tokens=args.max_new_tokens, stop_ids=stop):
        out.append(tok)
    t2 = time.perf_counter()
    raw = bytes(b for b in out if b < 256)
    print(json.dumps({
        "preset": preset,
        "forward_quant": args.forward_quant,
        "backward_mode": args.backward_mode,
        "prompt_bytes": len(prompt),
        "generated": out,
        "text_utf8": raw.decode("utf-8", errors="replace"),
        "prefill_s": t1 - t0,
        "decode_s": t2 - t1,
        "tokens_per_s": (len(out) / (t2 - t1)) if t2 > t1 else 0.0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
