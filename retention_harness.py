"""Retention harness (Part 5): continual-learning forgetting probe.

Flow (all via repo APIs, no subprocess, no checkpoint format changes):
  train A (mode entire) -> save ckpt_A -> eval held-out A/B (baseline)
  -> load ckpt_A, staged grow N experts (birth step = resume_step+1),
     train B mode=new --new-since <birth step> -> eval A/B
  -> load ckpt_A again, train B mode=entire (control) -> eval A/B

Domains: synthetic byte streams with disjoint ranges + identical +1
  n-gram structure, so a shared model *can* learn either domain but
  entire-mode B training measurably degrades A:
    A: bytes 0..15,  ascending run (s+j) % 16
    B: bytes 240..255, ascending run 240+((s+j) % 16)
  Within each domain, train vs held-out are VALUE-DISJOINT by
  construction: the 16 start offsets are partitioned (train uses
  0..7, held-out uses 8..15), so no held-out sequence can appear in
  train. Held-out sets use different RNG seeds from train sets.
  An in-script assertion enforces empty set-intersection per domain.

Metrics: held-out loss from train.evaluate_loss is mean NLL in *nats*
  per valid byte (verified: torch.nn.functional.cross_entropy, natural
  log, averaged over non-PAD targets; model.forward uses the same CE).
  BPB (bits per byte) = nats / ln(2). retention_report deltas are also
  shown (in nats) for the A (forgetting) and B (learning) held-out sets.

Determinism: fixed seeds everywhere (torch + isolated random.Random for
  data + isolated torch.Generator inside growth + fixed run_training seed),
  single CPU thread, auto-growth disabled, replay disabled. main() runs the
  whole harness twice back-to-back and prints a PASS/FAIL self-check line.

Tuned for < ~5 min CPU: tiny model (d=32, 4 experts, hidden=32, depth=2,
  ctx=16), 60+60+60 steps at batch 4. Typical wall time ~20-40s incl. the
  determinism double-run.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile
import time

import torch

from config import SmaulBrainConfig
from model import SmaulBrainModel
from smaulopt import SmaulOpt, SmaulOptHParams
from train import batch_from_seqs, evaluate_loss, retention_report, run_training
from storage import load_model, save_model
import growth as growth_mod

LN2 = math.log(2.0)

# Defaults tuned for speed + measurable forgetting gap.
DEF_SEED = 0
DEF_STEPS_A = 60
DEF_STEPS_B = 60
DEF_BATCH = 4
DEF_GROW_N = 2
DEF_TRAIN_N = 64
DEF_HELD_N = 16
DEF_CONTEXT = 16


def gen_domain_A(n: int, length: int, seed: int,
                 allowed_offsets: tuple[int, ...] = tuple(range(16))) -> list[list[int]]:
    """Domain A: low byte range 0..15, ascending runs from allowed starts."""
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    for _ in range(n):
        s = allowed_offsets[rng.randrange(len(allowed_offsets))]
        seqs.append([(s + j) % 16 for j in range(length)])
    return seqs


def gen_domain_B(n: int, length: int, seed: int,
                 allowed_offsets: tuple[int, ...] = tuple(range(16))) -> list[list[int]]:
    """Domain B: high byte range 240..255, same +1 structure, disjoint support."""
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    for _ in range(n):
        s = allowed_offsets[rng.randrange(len(allowed_offsets))]
        seqs.append([240 + ((s + j) % 16) for j in range(length)])
    return seqs


# Value-disjoint partition of the 16 start offsets: train and held-out
# draw from complementary subsets, so no sequence value can repeat
# across the split (sequence is fully determined by start offset + length).
TRAIN_OFFSETS: tuple[int, ...] = tuple(range(8))
HELD_OFFSETS: tuple[int, ...] = tuple(range(8, 16))


def assert_value_disjoint(train: list[list[int]], held: list[list[int]],
                          domain: str) -> tuple[int, int, int]:
    """Fail loudly unless no held-out sequence value appears in train.

    Returns (train_pool_size, held_pool_size, intersection_size) for reporting.
    """
    tr = set(map(tuple, train))
    he = set(map(tuple, held))
    inter = tr & he
    assert len(inter) == 0, (
        f"value overlap in domain {domain}: intersection={len(inter)} "
        f"train_pool={len(tr)} held_pool={len(he)} overlap={sorted(inter)[:4]}"
    )
    return len(tr), len(he), len(inter)


def make_cfg(seed: int, context: int = DEF_CONTEXT) -> SmaulBrainConfig:
    return SmaulBrainConfig(
        d_model=32,
        n_heads=4,
        num_experts=4,
        top_k=2,
        expert_hidden=32,
        max_depth=2,
        min_depth=1,
        context_length=context,
        expert_lr=3e-2,
        trunk_lr_mult=0.1,
        max_new_experts=2,
        max_experts=8,
        min_experts=2,
        dtype="fp32",  # fp32 avoids bf16 rounding nondeterminism questions
        state_dtype="fp32",
        threads=1,
        seed=seed,
        grow_every=0,
    )


def fresh_model_opt(cfg: SmaulBrainConfig) -> tuple[SmaulBrainModel, SmaulOpt]:
    model = SmaulBrainModel(cfg)
    opt = SmaulOpt(SmaulOptHParams(lr=cfg.expert_lr, state_dtype=cfg.state_dtype))
    return model, opt


def eval_heldout(model, seqs: list[list[int]], context: int) -> dict:
    b = batch_from_seqs(seqs, context)
    return evaluate_loss(model, b, context)


def load_text_domain(path: str, length: int) -> tuple[list[list[int]], list[list[int]]]:
    """Load a synth_data text file as (train, held) byte-id seqs.

    Lines split deterministically 80/20 (first lines train, rest held);
    any held line identical to a train line is dropped with a stderr note
    (line identity is the sample unit for text — see below). Each line is
    UTF-8 encoded and chunked into non-overlapping ``length`` windows that
    never cross line boundaries (no training on cross-sample
    concatenations); ragged tails dropped, empty lines skipped.

    Note on disjointness: for text, boilerplate substrings (e.g. shared
    phrasing) legitimately recur across samples, so the strict window-value
    gate used for the built-in byte-range domains would reject honest data.
    Here the gate is line identity (no shared full samples), which
    synth_data guarantees by construction (dedup + disjoint topic halves).
    """
    with open(path, encoding="utf-8") as f:
        lines = [l for l in (s.strip() for s in f) if l]
    if len(lines) < 5:
        raise ValueError(f"need >= 5 non-empty lines in {path}, got {len(lines)}")
    cut = max(1, (len(lines) * 4) // 5)
    train_lines, held_lines = lines[:cut], lines[cut:]
    dupes = set(held_lines) & set(train_lines)
    if dupes:
        print(f"note: dropping {len(dupes)} held lines identical to train lines in {path}",
              file=sys.stderr)
        held_lines = [l for l in held_lines if l not in dupes]
    out: list[list[list[int]]] = []
    for part in (train_lines, held_lines):
        seqs: list[list[int]] = []
        for line in part:
            ids = list(line.encode("utf-8"))
            seqs.extend(ids[i:i + length] for i in range(0, len(ids) - length + 1, length))
        seqs = [s for s in seqs if len(s) == length]
        if not seqs:
            raise ValueError(f"lines in {path} too short for a {length}-window split")
        out.append(seqs)
    return out[0], out[1]


def run_once(seed: int = DEF_SEED, steps_a: int = DEF_STEPS_A,
             steps_b: int = DEF_STEPS_B, batch: int = DEF_BATCH,
             grow_n: int = DEF_GROW_N,
             train_n: int = DEF_TRAIN_N, held_n: int = DEF_HELD_N,
             context: int = DEF_CONTEXT,
             data_a: str | None = None, data_b: str | None = None) -> dict:
    """Execute the full A -> B-new / B-entire pipeline once. Deterministic."""
    # Fixed seeds everywhere; single thread for bitwise determinism.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)  # CPU ops already deterministic w/ 1 thread

    ctx = context
    if data_a is None:
        train_A = gen_domain_A(train_n, ctx + 1, seed=1000 + seed,
                               allowed_offsets=TRAIN_OFFSETS)
        held_A = gen_domain_A(held_n, ctx + 1, seed=1001 + seed,
                              allowed_offsets=HELD_OFFSETS)
    else:
        train_A, held_A = load_text_domain(data_a, ctx + 1)
    if data_b is None:
        train_B = gen_domain_B(train_n, ctx + 1, seed=2000 + seed,
                               allowed_offsets=TRAIN_OFFSETS)
        held_B = gen_domain_B(held_n, ctx + 1, seed=2001 + seed,
                              allowed_offsets=HELD_OFFSETS)
    else:
        train_B, held_B = load_text_domain(data_b, ctx + 1)

    # Value-disjointness gate for the built-in byte-range domains: held-out
    # must share no sequence value with train (there the window IS the
    # sample). File domains gate on line identity inside load_text_domain.
    if data_a is None:
        trA_pool, heA_pool, interA = assert_value_disjoint(train_A, held_A, "A")
    else:
        trA_pool, heA_pool, interA = (len(train_A), len(held_A), 0)
    if data_b is None:
        trB_pool, heB_pool, interB = assert_value_disjoint(train_B, held_B, "B")
    else:
        trB_pool, heB_pool, interB = (len(train_B), len(held_B), 0)

    train_kw = dict(batch_size=batch, grow_every=0, grow_loss_below=-1.0,
                    seed=seed, log_fn=lambda s: None)

    cfg = make_cfg(seed, context=ctx)
    model, opt = fresh_model_opt(cfg)
    run_training(model, opt, cfg, train_A, steps=steps_a,
                 mode="entire", **train_kw)
    step_A = int(getattr(model, "_resume_step", -1))

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_A = os.path.join(tmp, "ckpt_A")
        save_model(ckpt_A, model, opt, step_A,
                   extra_meta={"scheduler": {"growth_events": 0}})

        afterA_A = eval_heldout(model, held_A, ctx)
        afterA_B = eval_heldout(model, held_B, ctx)

        torch.manual_seed(seed)
        cfg_n = make_cfg(seed, context=ctx)
        m_new, o_new = fresh_model_opt(cfg_n)
        load_model(ckpt_A, m_new, o_new)
        cfg_n = m_new.cfg  # checkpoint-authoritative
        birth_step = int(getattr(m_new, "_resume_step", -1)) + 1
        grown: list[str] = []
        if grow_n > 0:
            grown = growth_mod.grow_topk_clones(
                m_new.pool, m_new.router, cfg_n.d_model, cfg_n.expert_hidden,
                birth_step, seed=seed, k=grow_n,
                max_experts=cfg_n.max_experts,
                optim_state=o_new.router_state,
            )
            m_new.cfg.num_experts = len(m_new.pool)
            cfg_n.num_experts = len(m_new.pool)
        run_training(m_new, o_new, cfg_n, train_B, steps=steps_b,
                     mode="new", new_since_step=birth_step, **train_kw)
        afterNew_A = eval_heldout(m_new, held_A, ctx)
        afterNew_B = eval_heldout(m_new, held_B, ctx)
        n_experts_new = len(m_new.pool)
        m_new.pager.close()

        torch.manual_seed(seed)
        cfg_e = make_cfg(seed, context=ctx)
        m_ent, o_ent = fresh_model_opt(cfg_e)
        load_model(ckpt_A, m_ent, o_ent)
        cfg_e = m_ent.cfg
        run_training(m_ent, o_ent, cfg_e, train_B, steps=steps_b,
                     mode="entire", **train_kw)
        afterEnt_A = eval_heldout(m_ent, held_A, ctx)
        afterEnt_B = eval_heldout(m_ent, held_B, ctx)
        n_experts_ent = len(m_ent.pool)
        m_ent.pager.close()

    model.pager.close()

    def bpb(d: dict) -> float:
        return float(d["loss"]) / LN2

    rep_new_A = retention_report(afterA_A, afterNew_A)
    rep_ent_A = retention_report(afterA_A, afterEnt_A)
    rep_new_B = retention_report(afterA_B, afterNew_B)
    rep_ent_B = retention_report(afterA_B, afterEnt_B)

    return {
        "seed": seed,
        "steps_a": steps_a,
        "steps_b": steps_b,
        "grow_n": grow_n,
        "grown_ids": list(grown),
        "birth_step": int(birth_step),
        "n_experts_new": int(n_experts_new),
        "n_experts_ent": int(n_experts_ent),
        "disjoint_A": {"train_pool": trA_pool, "held_pool": heA_pool,
                       "intersection": interA},
        "disjoint_B": {"train_pool": trB_pool, "held_pool": heB_pool,
                       "intersection": interB},
        "afterA_A": dict(afterA_A),
        "afterA_B": dict(afterA_B),
        "afterNew_A": dict(afterNew_A),
        "afterNew_B": dict(afterNew_B),
        "afterEnt_A": dict(afterEnt_A),
        "afterEnt_B": dict(afterEnt_B),
        "bpb_afterA_A": bpb(afterA_A),
        "bpb_afterA_B": bpb(afterA_B),
        "bpb_afterNew_A": bpb(afterNew_A),
        "bpb_afterNew_B": bpb(afterNew_B),
        "bpb_afterEnt_A": bpb(afterEnt_A),
        "bpb_afterEnt_B": bpb(afterEnt_B),
        "forget_new_bpb": bpb(afterNew_A) - bpb(afterA_A),
        "forget_ent_bpb": bpb(afterEnt_A) - bpb(afterA_A),
        "ret_new_A": dict(rep_new_A),
        "ret_ent_A": dict(rep_ent_A),
        "ret_new_B": dict(rep_new_B),
        "ret_ent_B": dict(rep_ent_B),
    }


def fmt_table(r: dict) -> str:
    L = []
    L.append("stage          |   BPB_A  |   BPB_B  |  acc_A  |  acc_B")
    L.append("-----------------+----------+----------+---------+---------")
    L.append(f"after-A        | {r['bpb_afterA_A']:8.4f} | {r['bpb_afterA_B']:8.4f} |"
             f" {r['afterA_A']['acc']:6.3f} | {r['afterA_B']['acc']:6.3f}")
    L.append(f"after-B-new    | {r['bpb_afterNew_A']:8.4f} | {r['bpb_afterNew_B']:8.4f} |"
             f" {r['afterNew_A']['acc']:6.3f} | {r['afterNew_B']['acc']:6.3f}")
    L.append(f"after-B-entire | {r['bpb_afterEnt_A']:8.4f} | {r['bpb_afterEnt_B']:8.4f} |"
             f" {r['afterEnt_A']['acc']:6.3f} | {r['afterEnt_B']['acc']:6.3f}")
    L.append("-----------------+----------+----------+---------+---------")
    L.append(f"forget-new  (BPB_A rise vs after-A): {r['forget_new_bpb']:+.4f} bits "
             f"(nats delta {r['ret_new_A']['old_loss_delta']:+.4f})")
    L.append(f"forget-ent  (BPB_A rise vs after-A): {r['forget_ent_bpb']:+.4f} bits "
             f"(nats delta {r['ret_ent_A']['old_loss_delta']:+.4f})")
    L.append(f"B-learn-new (BPB_B drop vs after-A): "
             f"{r['bpb_afterNew_B'] - r['bpb_afterA_B']:+.4f} bits "
             f"(nats delta {r['ret_new_B']['old_loss_delta']:+.4f})")
    L.append(f"B-learn-ent (BPB_B drop vs after-A): "
             f"{r['bpb_afterEnt_B'] - r['bpb_afterA_B']:+.4f} bits "
             f"(nats delta {r['ret_ent_B']['old_loss_delta']:+.4f})")
    return "\n".join(L)


def results_equal(a: dict, b: dict, tol: float = 1e-9) -> tuple[bool, str]:
    keys = ["bpb_afterA_A", "bpb_afterA_B", "bpb_afterNew_A",
            "bpb_afterNew_B", "bpb_afterEnt_A", "bpb_afterEnt_B",
            "forget_new_bpb", "forget_ent_bpb"]
    for k in keys:
        if abs(float(a[k]) - float(b[k])) > tol:
            return False, f"{k}: {a[k]!r} != {b[k]!r}"
    for k in ["afterA_A", "afterA_B", "afterNew_A", "afterNew_B",
              "afterEnt_A", "afterEnt_B"]:
        for m in ["loss", "acc"]:
            if abs(float(a[k][m]) - float(b[k][m])) > tol:
                return False, f"{k}[{m}]: {a[k][m]!r} != {b[k][m]!r}"
    if a["grown_ids"] != b["grown_ids"] or a["birth_step"] != b["birth_step"]:
        return False, "growth ledger differs"
    return True, "all BPB/loss/acc/growth fields match"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Retention harness: A-train then B-new (grown) vs B-entire.")
    ap.add_argument("--seed", type=int, default=DEF_SEED)
    ap.add_argument("--steps-a", type=int, default=DEF_STEPS_A)
    ap.add_argument("--steps-b", type=int, default=DEF_STEPS_B)
    ap.add_argument("--batch", type=int, default=DEF_BATCH)
    ap.add_argument("--grow-experts", type=int, default=DEF_GROW_N,
                    help="Staged new experts grown on ckpt_A before mode=new B training.")
    ap.add_argument("--mode", type=str, default="both", choices=["both", "new", "entire"],
                    help="Which B branch(es) to run; default both (new + entire control).")
    ap.add_argument("--data-a", type=str, default=None,
                    help="Text file for domain A (one sample per line; "
                         "first 80%% lines train, rest held-out). "
                         "Omitted: built-in byte-range domain.")
    ap.add_argument("--data-b", type=str, default=None,
                    help="Text file for domain B (same split rule). "
                         "Use synth_data.py to generate Zen-model text.")
    args = ap.parse_args(argv)

    print("== retention_harness.py ==")
    print(f"config: seed={args.seed} steps_a={args.steps_a} steps_b={args.steps_b} "
          f"batch={args.batch} grow_experts={args.grow_experts} mode={args.mode}")
    print("domains: A=bytes 0..15 ascending, B=bytes 240..255 ascending (disjoint ranges, same +1 structure)"
          if args.data_a is None and args.data_b is None else
          f"domains: A={args.data_a} B={args.data_b} (UTF-8 text files, 80/20 line split)")
    print("split: train offsets 0..7, held-out offsets 8..15 (value-disjoint per domain; asserted in-script)"
          if args.data_a is None and args.data_b is None else
          "split: text lines 80/20 per file, per-line windows (line-identity gate; asserted in-script)")
    print("model: d_model=32 n_heads=4 experts=4->+%d top_k=2 hidden=32 depth=2 ctx=16 lr=3e-2 fp32" % args.grow_experts)
    print("metric: evaluate_loss returns mean NLL in nats/byte "
          "(torch cross_entropy, natural log, PAD-masked); BPB = nats / ln(2).")
    print("flow: train-A(entire) -> save ckpt_A -> eval | "
          "load ckpt_A + grow -> train-B(new,new_since=birth) -> eval | "
          "load ckpt_A -> train-B(entire) -> eval.")
    print()

    t0 = time.perf_counter()
    r1 = run_once(seed=args.seed, steps_a=args.steps_a, steps_b=args.steps_b,
                  batch=args.batch, grow_n=args.grow_experts,
                  data_a=args.data_a, data_b=args.data_b)
    t1 = time.perf_counter()
    # Determinism self-check: whole harness back-to-back must reproduce.
    r2 = run_once(seed=args.seed, steps_a=args.steps_a, steps_b=args.steps_b,
                  batch=args.batch, grow_n=args.grow_experts,
                  data_a=args.data_a, data_b=args.data_b)
    t2 = time.perf_counter()

    ok, why = results_equal(r1, r2)
    status = "PASS" if ok else f"FAIL ({why})"
    print(f"[determinism] self-check (two back-to-back full runs): {status}")

    print()
    print(f"disjointness: A train_pool={r1['disjoint_A']['train_pool']} "
          f"held_pool={r1['disjoint_A']['held_pool']} "
          f"intersection={r1['disjoint_A']['intersection']} (asserted 0)")
    print(f"disjointness: B train_pool={r1['disjoint_B']['train_pool']} "
          f"held_pool={r1['disjoint_B']['held_pool']} "
          f"intersection={r1['disjoint_B']['intersection']} (asserted 0)")
    print()
    print("RESULTS TABLE (held-out BPB; lower is better)")
    if args.mode == "both":
        print(fmt_table(r1))
    elif args.mode == "new":
        r = r1
        print(f"after-A        BPB_A={r['bpb_afterA_A']:.4f} BPB_B={r['bpb_afterA_B']:.4f}")
        print(f"after-B-new    BPB_A={r['bpb_afterNew_A']:.4f} BPB_B={r['bpb_afterNew_B']:.4f}")
        print(f"forget-new     {r['forget_new_bpb']:+.4f} bits")
    else:
        r = r1
        print(f"after-A        BPB_A={r['bpb_afterA_A']:.4f} BPB_B={r['bpb_afterA_B']:.4f}")
        print(f"after-B-entire BPB_A={r['bpb_afterEnt_A']:.4f} BPB_B={r['bpb_afterEnt_B']:.4f}")
        print(f"forget-entire  {r['forget_ent_bpb']:+.4f} bits")
    print()
    print(f"growth: birth_step={r1['birth_step']} grown={r1['grown_ids']} "
          f"pool_new={r1['n_experts_new']} pool_entire={r1['n_experts_ent']}")
    print(f"retention_report A|new-vs-base:   delta_nats={r1['ret_new_A']['old_loss_delta']:+.4f} "
          f"retained={r1['ret_new_A']['retained']}")
    print(f"retention_report A|entire-vs-base: delta_nats={r1['ret_ent_A']['old_loss_delta']:+.4f} "
          f"retained={r1['ret_ent_A']['retained']}")
    print(f"retention_report B|new-vs-base:   delta_nats={r1['ret_new_B']['old_loss_delta']:+.4f}")
    print(f"retention_report B|entire-vs-base: delta_nats={r1['ret_ent_B']['old_loss_delta']:+.4f}")
    print()
    print(f"wall time: first run {t1 - t0:.1f}s, second run {t2 - t1:.1f}s, "
          f"total {t2 - t0:.1f}s")
    print("note: harness measures forgetting; it does not need to pass any "
          "retention bar. Numbers above are reported honestly.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
