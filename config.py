"""Central architecture + training configuration for SmaulBRAIN.

Single source of truth for model topology, precision policy, paging,
adaptive halting, optimizer groups, and CLI defaults. A checkpoint stores
a serialized copy of this config so topology changes stay checkpoint-safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


__version__ = "0.1.0"


# Expert sizing: a SwiGLU expert with d_model=d and hidden=h owns
# 3*d*h parameters (gate, up, down projections, no biases).
# At d=512, h=3328 -> 3*512*3328 = 5,111,808 ~= 5.12M.
EXPERT_TARGET_PARAMS = 5_120_000


def expert_param_count(d_model: int, expert_hidden: int) -> int:
    """Exact parameter count of one SwiGLU expert (no biases)."""
    return 3 * d_model * expert_hidden


def expert_hidden_for_target(d_model: int, target: int = EXPERT_TARGET_PARAMS) -> int:
    """Hidden dim giving ~target params for a given d_model (multiple of 32)."""
    h = max(32, round(target / (3 * d_model) / 32) * 32)
    return int(h)


@dataclass
class SmaulBrainConfig:
    """Full SmaulBRAIN configuration. All CLI flags map onto these fields."""

    # --- core topology (tiny/CPU-friendly defaults; see --full in cli.py) ---
    vocab_size: int = 256  # raw bytes; specials extend this (see bytes.py)
    d_model: int = 64
    n_heads: int = 4
    # --- recurrent block (shared, applied min_depth..max_depth times) ---
    min_depth: int = 1
    max_depth: int = 3
    # --- adaptive halting (PonderNet-style) ---
    halting_threshold: float = 0.9
    halt_prior: float = 0.1  # geometric prior p for ponder KL
    ponder_beta: float = 0.01  # weight of ponder KL regularizer
    # --- dynamic MoE ---
    num_experts: int = 8
    top_k: int = 2
    active_experts: int = 2  # alias enforced == top_k at runtime
    max_experts: int = 64
    min_experts: int = 2
    expert_hidden: int = 128  # explicit knob; full preset uses 3328 for ~5.12M
    moe_balance_weight: float = 0.01
    capacity_factor: float = 1.5
    # --- context / sequence ---
    context_length: int = 128
    attention_chunk_size: int = 256  # causal linear-attention training chunk
    # --- paging ---
    paging_method: str = "D2R"  # D2R | R2VR | D2VR
    ram_cache: int = 8  # max experts resident in RAM cache
    vram_cache: int = 4  # max experts resident in VRAM cache
    # --- optimization ---
    expert_lr: float = 2e-4
    trunk_lr_mult: float = 0.1  # shared trunk LR = expert_lr * trunk_lr_mult
    router_lr_mult: float = 1.0
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta_m: float = 0.9
    beta_v: float = 0.999
    epsilon: float = 1e-8
    # --- precision policy ---
    dtype: str = "bf16"  # compute dtype for activations: bf16 | fp32
    fp8_tile: int = 64  # block size for FP8 per-block scaling
    state_dtype: str = "bf16"  # optimizer state storage: bf16 | fp32
    # --- misc ---
    threads: int = 2
    seed: int = 0
    rmsnorm_eps: float = 1e-6
    # --- growth / pruning ---
    grow_every: int = 200  # optimizer steps between growth evaluations
    prune_survival_steps: int = 500  # grace period before an expert may die
    prune_min_usage: float = 1e-4  # usage share below which expert is dying
    max_new_experts: int = 2  # cap per growth event (reproducibility)

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must split over heads"
        assert 1 <= self.min_depth <= self.max_depth, "need 1 <= min <= max depth"
        assert 0.0 < self.halting_threshold < 1.0, "halting threshold in (0,1)"
        assert 0.0 < self.halt_prior < 1.0, "halt prior in (0,1)"
        assert 1 <= self.top_k <= self.num_experts, "need 1 <= top_k <= experts"
        assert self.num_experts >= 1 and self.max_experts >= self.num_experts
        assert self.min_experts >= self.top_k and self.min_experts <= self.num_experts
        assert self.paging_method in ("D2R", "R2VR", "D2VR"), "bad paging method"
        assert self.ram_cache >= 1 and self.vram_cache >= 1, "caches need >= 1 slot"
        assert self.dtype in ("bf16", "fp32"), "compute dtype bf16|fp32"
        assert self.state_dtype in ("bf16", "fp32"), "state dtype bf16|fp32"
        assert self.expert_hidden >= 8, "expert hidden dim too small"
        assert self.attention_chunk_size >= 1, "attention chunk size must be positive"
        # active_experts is a user-facing alias for top_k; keep them in sync.
        object.__setattr__(self, "active_experts", self.top_k)

    # --- derived counts ---
    @property
    def trunk_lr(self) -> float:
        return self.expert_lr * self.trunk_lr_mult

    @property
    def router_lr(self) -> float:
        return self.expert_lr * self.router_lr_mult

    @property
    def per_expert_params(self) -> int:
        return expert_param_count(self.d_model, self.expert_hidden)

    def shared_params(self) -> int:
        """Shared trunk params: embeddings + attention + norms + halt + router + head."""
        d, v = self.d_model, self.vocab_size
        # embed(v,d) + qkv+o (4*d*d) + 6 norms (n_init,n1,n_attn,n2,n3,n_final)
        # + halt (d+1) + router (n_exp*d + n_exp) + out head (v*d)
        return v * d + 4 * d * d + 6 * d + (d + 1) + (self.num_experts * d + self.num_experts) + v * d

    def total_params(self) -> int:
        return self.shared_params() + self.num_experts * self.per_expert_params

    def active_params(self) -> int:
        return self.shared_params() + self.top_k * self.per_expert_params

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SmaulBrainConfig":
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in d.items() if k in known}
        # Older checkpoints allowed an impossible pruning floor below top-k.
        # Clamp it to the minimum viable active expert count when loading.
        values["min_experts"] = max(
            int(values.get("min_experts", cls.min_experts)),
            int(values.get("top_k", cls.top_k)),
        )
        return cls(**values)

    def describe_counts(self) -> dict:
        return {
            "shared_params": self.shared_params(),
            "router_params": self.num_experts * self.d_model + self.num_experts,
            "per_expert_params": self.per_expert_params,
            "expert_count": self.num_experts,
            "expert_params_total": self.num_experts * self.per_expert_params,
            "total_params": self.total_params(),
            "active_params": self.active_params(),
        }
