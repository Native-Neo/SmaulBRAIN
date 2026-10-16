"""Central configuration + immutable hardcoded presets.

Architecture family (all presets):
  - Byte vocab V=259 (256 bytes + BOS/EOS/PAD), tied embedding/head.
  - Per layer: 2x RMSNorm (d params each), recurrent attention
    (W_r,W_k,W_v,W_o: d*d each; decay w: d; bonus u: d),
    gated channel-mixing FFN (W_a,W_b: d*hff; W_c: hff*d, no biases).
  - Final RMSNorm (d). Tied head (no extra params).

Per-layer params = 4*d*d + 2*d (decay+bonus) + 2*d (norms)
                   + 3*d*hff = 4*d*d + 4*d + 3*d*hff.
Total = V*d + d + L*(4*d*d + 4*d + 3*d*hff)   [tied head]

MoE experts: 2-matrix MLP with GELU: W_in (d*h), W_out (h*d),
  per-expert = 2*d*h. Router: d * n_experts.
"""
from dataclasses import dataclass, field

from . import vocab as _vocab

GROWTH_DEFAULT_EVERY_STEPS = 10000
GROWTH_DEFAULT_LOSS_NATS = 0.5

PAGING_MODES = ("R", "V", "D2R", "D2V", "R2V")
FORWARD_QUANT_CHOICES = ("fp32", "fp8", "ternary")
BACKWARD_MODE_CHOICES = ("fp32", "fp8", "ternary")

# (forward, backward) pairs explicitly supported.
SUPPORTED_QUANT_COMBOS = {
    ("fp32", "fp32"),
    ("fp8", "fp8"),
    ("ternary", "fp32"),
    ("ternary", "ternary"),
    ("ternary", "fp8"),
}

PRESET_ALIASES = {
    "full": None,  # resolved only together with --moe/--dense
    "tiny": None,
    "moe-full": "moe-full",
    "moe-tiny": "moe-tiny",
    "dense-full": "dense-full",
    "dense-tiny": "dense-tiny",
}

VALID_PRESETS = ("moe-full", "moe-tiny", "dense-full", "dense-tiny")


@dataclass(frozen=True)
class PresetSpec:
    name: str
    kind: str  # "moe" | "dense"
    d_model: int
    n_layers: int
    ffn_hidden: int
    n_experts_init: int = 0
    top_k: int = 0
    expert_hidden: int = 0
    tied_embeddings: bool = True
    vocab_size: int = _vocab.VOCAB_SIZE


# Exact-count presets. Verified by arithmetic in tests/test_presets.py.
# dense-tiny: 259*32+32 + 15*(4*32*32+4*32+3*32*128) = 8320+247680 = 256000
# dense-full: 259*1000+1000 + 19*(4*1000*1000+4*1000+3*1000*3152) = 260000+255740000 = 256000000
# moe-full expert: 2*512*5000 = 5120000 ; moe-tiny expert: 2*64*80 = 10240
PRESETS: dict = {
    "dense-tiny": PresetSpec(name="dense-tiny", kind="dense", d_model=32,
                             n_layers=15, ffn_hidden=128),
    "dense-full": PresetSpec(name="dense-full", kind="dense", d_model=1000,
                             n_layers=19, ffn_hidden=3152),
    "moe-full": PresetSpec(name="moe-full", kind="moe", d_model=512,
                           n_layers=6, ffn_hidden=0,
                           n_experts_init=64, top_k=8, expert_hidden=5000),
    "moe-tiny": PresetSpec(name="moe-tiny", kind="moe", d_model=64,
                           n_layers=4, ffn_hidden=0,
                           n_experts_init=32, top_k=2, expert_hidden=80),
}

EXPECTED_EXPERT_PARAMS = {"moe-full": 5_120_000, "moe-tiny": 10_240}
EXPECTED_DENSE_TOTAL = {"dense-full": 256_000_000, "dense-tiny": 256_000}
GROWTH_ADD_FULL = 8
GROWTH_ADD_TINY = 2


def dense_param_total(spec: PresetSpec) -> int:
    d, h, L, V = spec.d_model, spec.ffn_hidden, spec.n_layers, spec.vocab_size
    per_layer = 4 * d * d + 4 * d + 3 * d * h
    return V * d + d + L * per_layer


def expert_param_count(spec: PresetSpec) -> int:
    return 2 * spec.d_model * spec.expert_hidden


def shared_param_total(spec: PresetSpec) -> int:
    """Non-expert trainable params of an MoE model (embed+recurrent+norms+router)."""
    d, L, V = spec.d_model, spec.n_layers, spec.vocab_size
    per_layer = 4 * d * d + 4 * d  # attention+decay+bonus+2 norms, no FFN (experts replace it)
    # each MoE layer keeps a small gate projection d*d for expert-input mixing? No:
    # channel mixing lives inside experts, so shared per-layer has no FFN params.
    embed = V * d
    final_norm = d
    router = d * spec.n_experts_init
    return embed + final_norm + L * per_layer + router


def whole_model_total(spec: PresetSpec, n_experts=None) -> int:
    if spec.kind == "dense":
        return dense_param_total(spec)
    n = spec.n_experts_init if n_experts is None else n_experts
    return shared_param_total(spec) + n * expert_param_count(spec)


def resolve_preset(cli_preset: str, moe: bool, dense: bool) -> str:
    if moe and dense:
        raise ValueError("--moe and --dense are mutually exclusive")
    p = (cli_preset or "").strip().lower()
    if p in ("full", "tiny"):
        if moe and not dense:
            return "moe-" + p
        if dense and not moe:
            return "dense-" + p
        raise ValueError(
            f"--preset '{cli_preset}' requires exactly one of --moe or --dense")
    if p in VALID_PRESETS:
        want = p.split("-")[0]
        if moe and want == "dense":
            raise ValueError(f"--moe conflicts with preset '{p}' (dense preset)")
        if dense and want == "moe":
            raise ValueError(f"--dense conflicts with preset '{p}' (moe preset)")
        return p
    raise ValueError(f"invalid preset '{cli_preset}'; expected one of "
                     f"full|tiny (with --moe/--dense) or {', '.join(VALID_PRESETS)}")


def parse_growth_every_steps(raw) -> int:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"--growth-every-steps must be a non-negative integer, got {raw!r}")
    if v < 0:
        raise ValueError(f"--growth-every-steps must be >= 0, got {v}")
    return v


def parse_growth_loss_nats(raw):
    if raw is None:
        return GROWTH_DEFAULT_LOSS_NATS
    if isinstance(raw, str) and raw.strip().lower() == "none":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"--growth-loss-nats must be a finite non-negative number or 'none', got {raw!r}")
    if not (v >= 0) or v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"--growth-loss-nats must be a finite non-negative number or 'none', got {raw!r}")
    return v


def validate_quant_combo(forward_quant: str, backward_mode: str) -> None:
    if forward_quant not in FORWARD_QUANT_CHOICES:
        raise ValueError(f"--forward-quant must be one of {FORWARD_QUANT_CHOICES}")
    if backward_mode not in BACKWARD_MODE_CHOICES:
        raise ValueError(f"--backward-mode must be one of {BACKWARD_MODE_CHOICES}")
    if (forward_quant, backward_mode) not in SUPPORTED_QUANT_COMBOS:
        raise ValueError(
            f"unsupported quantization combination forward={forward_quant} "
            f"backward={backward_mode}; supported: {sorted(SUPPORTED_QUANT_COMBOS)}")


def arch_report(spec: PresetSpec, n_experts_current=None) -> dict:
    d = spec.d_model
    rep = {
        "preset": spec.name,
        "kind": spec.kind,
        "d_model": d,
        "n_layers": spec.n_layers,
        "ffn_hidden": spec.ffn_hidden,
        "vocab_size": spec.vocab_size,
        "recurrent_state_per_layer": d,
        "recurrent_state_total": spec.n_layers * d,
        "recurrent_state_bytes_fp32": spec.n_layers * d * 4,
    }
    if spec.kind == "dense":
        rep["total_params"] = dense_param_total(spec)
    else:
        n = spec.n_experts_init if n_experts_current is None else n_experts_current
        rep["experts_init"] = spec.n_experts_init
        rep["experts_current"] = n
        rep["top_k"] = spec.top_k
        rep["expert_hidden"] = spec.expert_hidden
        rep["per_expert_params"] = expert_param_count(spec)
        rep["shared_params"] = shared_param_total(spec)
        rep["router_params"] = d * n
        rep["total_params"] = shared_param_total(spec) + n * expert_param_count(spec)
    return rep
