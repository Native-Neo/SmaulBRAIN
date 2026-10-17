#!/usr/bin/env python3
"""SmaulBRAIN V2 inference entry point (training-independent path)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smaulbrain import cli as _cli
from smaulbrain import config as _cfg
from smaulbrain import inference as _inf
from smaulbrain import model as _model


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run SmaulBRAIN V2 inference")
    _cli.add_common_args(ap)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--stop-eos", action="store_true")
    args = ap.parse_args(argv)
    try:
        preset = _cli.validate_args(args)
    except ValueError as e:
        ap.error(str(e))
    spec = _cfg.PRESETS[preset]
    man_path = os.path.join(args.checkpoint, "manifest.json")
    if not os.path.exists(man_path):
        print(f"error: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 2
    from smaulbrain import checkpoint as _ckpt
    blob = _ckpt.load_checkpoint(args.checkpoint)
    if blob["manifest"]["preset"] != preset:
        print(f"error: checkpoint preset {blob['manifest']['preset']} != {preset}",
              file=sys.stderr)
        return 2
    store = None
    if spec.kind == "moe":
        from smaulbrain import paging as _paging
        n_exp = blob["manifest"]["n_experts_current"]
        if args.paging in ("R", "V", "R2V"):
            ram = {e: {"Win": blob["params"][f"expert.{e}.Win"],
                       "Wout": blob["params"][f"expert.{e}.Wout"]} for e in range(n_exp)}
            store = _paging.ExpertStore(args.paging, n_exp, ram_weights=ram)
        else:
            import tempfile
            dd = tempfile.mkdtemp(prefix="smaul-infer-")
            import torch
            for e in range(n_exp):
                torch.save({"Win": blob["params"][f"expert.{e}.Win"].detach().cpu(),
                            "Wout": blob["params"][f"expert.{e}.Wout"].detach().cpu()},
                           os.path.join(dd, f"expert_{e}.pt"))
            store = _paging.ExpertStore(args.paging, n_exp, disk_dir=dd)
    m = _model.Model(spec, dict(blob["params"]), forward_quant=args.forward_quant,
                     backward_mode=args.backward_mode, paging_mode=args.paging,
                     paging_store=store, device=args.device)
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
