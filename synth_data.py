"""Synthetic training data generator via OpenCode Zen free models.

No API keys, no paid models: every call goes through the local
``opencode run`` CLI with a model from the Zen FREE allowlist below
(authenticate once with ``opencode auth login`` and pick Zen).
The generator model is user-chosen via ``--model`` but restricted to
free models only — anything else is refused.

Typical use (two disjoint domains for the A/B retention gauntlet)::

    python synth_data.py --preset AB --model big-pickle --out data/ab
    python cli.py --ckpt ckpt/demo --data data/ab/domainA_train.txt train --steps 20
    python retention_harness.py --data-a data/ab/domainA.txt --data-b data/ab/domainB.txt

Output layout (``--out DIR``)::

    DIR/<name>_train.txt     one sample per line (byte-safe text)
    DIR/<name>_held.txt      held-out samples, value-disjoint from train
    DIR/manifest.json        model, prompts, seeds, counts, sha256 per file

Value-disjointness is enforced by construction: train and held-out are
generated from disjoint topic pools, deduplicated, and any overlapping
line is dropped (with a loud stderr count, never silently kept).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

# Zen free text models only (chat/completions or responses endpoints that
# return assistant text). `jev-1.13-free` is intentionally excluded: it is
# a structured-decision endpoint, not free text. Pricing source:
# https://opencode.ai/docs/zen/ (all listed Free/Free/Free).
FREE_MODELS = (
    "big-pickle",
    "space-bunny-free",
    "longcat-2.5-preview-free",
    "step-5-preview-free",
    "exo-free",
    "fledge-alpha-free",
    "mimo-v2.6-flash-free",
    "mimo-v2.5-free",
    "ling-3.1-flash-free",
    "ling-3.0-flash-fin-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "muse-spark-1.3-contributor-free",
)

DEFAULT_MODEL = "big-pickle"
PROVIDER_PREFIX = "opencode/"

# Built-in gauntlet presets: disjoint everyday topics so domains A/B share
# almost no vocabulary (mirrors the byte-range split of the harness).
PRESETS = {
    "AB": {
        "domainA": {
            "label": "syslogs",
            "topics": [
                "disk failure SMART warnings", "kernel oops traces",
                "nginx access spikes", "postgres slow queries",
                "cron job failures", "TLS certificate expiry",
                "OOM killer events", "NTP clock drift",
            ],
            "style": "terse Unix syslog lines with timestamps, hostnames, and PIDs",
        },
        "domainB": {
            "label": "recipes",
            "topics": [
                "sourdough bread", "miso soup", "ratatouille",
                "pancakes", "lentil curry", "apple pie",
                "caesar salad", "fried rice",
            ],
            "style": "short home-cooking recipe steps with quantities",
        },
    },
}

PROMPT_TEMPLATE = (
    "Do not use any tools. Reply with ONLY the requested lines, no headers, "
    "no numbering, no commentary, no code fences.\n"
    "Write {n} distinct one-line {style} about '{topic}'. "
    "Each line 40-120 characters, plain ASCII, no blank lines. "
    "Batch {batch} of {batches}, avoid repeating earlier batches."
)


def normalize_model(name: str) -> str:
    """Accept `big-pickle` or `opencode/big-pickle`; refuse anything not free."""
    short = name[len(PROVIDER_PREFIX):] if name.startswith(PROVIDER_PREFIX) else name
    if short not in FREE_MODELS:
        raise ValueError(
            f"refusing non-free model {name!r}: choose one of "
            f"{', '.join(FREE_MODELS)} (see --list-models)"
        )
    return PROVIDER_PREFIX + short


def list_models() -> str:
    return "\n".join(f"  {PROVIDER_PREFIX}{m}" for m in FREE_MODELS)


def build_prompt(topic: str, style: str, n: int, batch: int, batches: int) -> str:
    return PROMPT_TEMPLATE.format(n=n, topic=topic, style=style,
                                  batch=batch, batches=batches)


def extract_text(events: str) -> str:
    """Pull assistant text out of `opencode run --format json` output.

    Tolerant by design (schema may evolve): walks every JSON object found
    per line and collects `part` dicts of type text. Non-JSON lines are
    ignored. Returns concatenated text (may be empty on tool-only runs).
    """
    chunks: list[str] = []
    for line in events.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if cur.get("type") == "text" and isinstance(cur.get("text"), str):
                    chunks.append(cur["text"])
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    return "".join(chunks)


def run_opencode(prompt: str, model: str, timeout: int,
                 runner=subprocess.run) -> str:
    """One non-interactive `opencode run` call; returns assistant text."""
    proc = runner(
        ["opencode", "run", "--model", model, "--format", "json", prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"opencode run failed (exit {proc.returncode}): "
            f"{(proc.stderr or '')[-500:]}"
        )
    text = extract_text(proc.stdout or "")
    if not text.strip():
        raise RuntimeError("opencode run returned no assistant text")
    return text


def clean_lines(text: str) -> list[str]:
    """Split, strip ASCII-printable lines, drop empties/fences/numbering."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        if not line or line.startswith("```"):
            continue
        # Strip leading "1. "/"- " list markers the model may add anyway.
        while line[:2] in ("- ", "* ") or (
                len(line) > 2 and line[0].isdigit()
                and line[1:3] in (". ", ") ")):
            line = line[3:] if line[1:3] in (". ", ") ") else line[2:]
            line = line.strip()
        line = "".join(c for c in line if 32 <= ord(c) < 127)
        if line:
            out.append(line)
    return out


def generate_domain(label: str, topics: list[str], style: str, model: str,
                    n_train: int, n_held: int, seed: int, timeout: int,
                    per_call: int = 8, retries: int = 3,
                    runner=subprocess.run, log_fn=None) -> tuple[list[str], list[str]]:
    """Generate train + value-disjoint held-out lines for one domain.

    Train draws from the first half of topics, held-out from the second
    half; leftovers are topped up round-robin. Exact duplicates and any
    train/held overlap are dropped (overlap counted to stderr).
    """
    log = log_fn or (lambda s: print(s, file=sys.stderr))
    rng = random.Random(seed)
    half = max(1, len(topics) // 2)
    plan = [("train", topics[:half], n_train), ("held", topics[half:], n_held)]
    got: dict[str, list[str]] = {"train": [], "held": []}
    seen: set[str] = set()
    for split, pool, want in plan:
        batches = max(1, (want + per_call - 1) // per_call)
        for b in range(batches):
            topic = pool[(b + rng.randrange(len(pool))) % len(pool)]
            prompt = build_prompt(topic, style, per_call, b + 1, batches)
            for attempt in range(retries):
                try:
                    lines = clean_lines(run_opencode(prompt, model, timeout, runner))
                    break
                except (RuntimeError, subprocess.TimeoutExpired) as e:
                    if attempt == retries - 1:
                        raise
                    log(f"retry {attempt + 1}/{retries} ({split} batch {b + 1}): {e}")
                    time.sleep(2 ** attempt)
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    got[split].append(line)
                if len(got[split]) >= want:
                    break
            if len(got[split]) >= want:
                break
        # Top-up round-robin if the model under-delivered.
        t = 0
        while len(got[split]) < want and t < want * 2:
            topic = pool[t % len(pool)]
            lines = clean_lines(run_opencode(
                build_prompt(topic, style, per_call, batches + t + 1, batches + t + 1),
                model, timeout, runner))
            added = False
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    got[split].append(line)
                    added = True
                    if len(got[split]) >= want:
                        break
            if not added:
                break
            t += 1
    train, held = got["train"][:n_train], got["held"][:n_held]
    overlap = set(train) & set(held)
    if overlap:
        log(f"dropping {len(overlap)} train/held overlapping lines")
        held = [l for l in held if l not in set(train)]
    return train, held


def write_domain(out_dir: str, name: str, train: list[str], held: list[str]) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split, lines in (("train", train), ("held", held)):
        path = os.path.join(out_dir, f"{name}_{split}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        paths[split] = {"path": path, "lines": len(lines), "sha256": digest}
    return paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synth_data",
                                description="Generate synthetic training text via Zen free models.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Zen free model (default: {DEFAULT_MODEL}). See --list-models.")
    p.add_argument("--list-models", action="store_true",
                   help="Print the free-model allowlist and exit.")
    p.add_argument("--preset", default="AB", choices=list(PRESETS),
                   help="Built-in domain pair (default: AB).")
    p.add_argument("--out", default="data/ab", help="Output directory.")
    p.add_argument("--n-train", type=int, default=64, help="Lines per domain train split.")
    p.add_argument("--n-held", type=int, default=16, help="Lines per domain held-out split.")
    p.add_argument("--seed", type=int, default=0, help="Topic rotation seed.")
    p.add_argument("--timeout", type=int, default=300, help="Seconds per opencode call.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts and plan without calling opencode.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_models:
        print(list_models())
        return 0
    try:
        model = normalize_model(args.model)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    preset = PRESETS[args.preset]
    if args.dry_run:
        for key, dom in preset.items():
            print(f"[{key}] {dom['label']}: {len(dom['topics'])} topics")
            print("  e.g. " + build_prompt(dom["topics"][0], dom["style"], 8, 1, 8)[:160] + "...")
        print(f"model={model} out={args.out} n_train={args.n_train} n_held={args.n_held}")
        return 0
    manifest: dict = {"model": model, "preset": args.preset, "seed": args.seed,
                      "domains": {}}
    for key, dom in preset.items():
        train, held = generate_domain(dom["label"], dom["topics"], dom["style"],
                                      model, args.n_train, args.n_held,
                                      args.seed, args.timeout)
        manifest["domains"][key] = {"label": dom["label"], **write_domain(
            args.out, key, train, held)}
        print(f"[{key}] train={len(train)} held={len(held)}", file=sys.stderr)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"model": model, "out": args.out,
                      "domains": {k: {"train": v["train"]["lines"], "held": v["held"]["lines"]}
                                  for k, v in manifest["domains"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
