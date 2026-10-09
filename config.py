"""Central architecture + training configuration for SmaulBRAIN.

Single source of truth for model topology, precision policy, paging,
adaptive halting, optimizer groups, and CLI defaults. A checkpoint stores
a serialized copy of this config so topology changes stay checkpoint-safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from bytes import total_vocab


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
    # Vocabulary covers raw bytes PLUS structural specials (bos/eos/pad/sep):
    # the model must accept special ids without index errors, and padding
    # uses PAD_ID (masked from the loss), never byte 0 (a real NUL byte).
    vocab_size: int = total_vocab()
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
    min_experts: int = 64  # pool floor respected by the offline pruner
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
    router_lr_mult: float = 1.0  # router LR = expert_lr * router_lr_mult
    # Old-expert router-row LR scale applied ONLY in training mode `new`:
    # applied old-row deltas for experts with birth_step < new_since_step
    # are rescaled by this factor post-step, newborn rows train at full LR.
    router_lr_mult_new: float = 0.005
    # Additive router-logit bonus for NEW experts (birth_step >= the run's
    # new_since_step cutoff), applied only while a runtime routing attribute
    # is set (mode `new` training wires it per train_step; see routing.py).
    # 0.0 = off (bit-exact no-op). Steering only changes routing
    # weights/choice, never old-expert weights.
    new_routing_bias: float = 0.0
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
    # --- growth (training) / pruning (manual offline tool) ---
    # Defaults encode the standing policy: grow every 20K steps or on
    # sub-1.0 loss. Training never prunes; `python pruning.py --rm-worst N`
    # removes the worst experts down to min_experts (honored from the
    # checkpoint's resume_config.json). prune_survival_steps/prune_min_usage
    # are the offline pruner's grace period and dying bar.
    grow_every: int = 20000  # optimizer steps between growth evaluations
    prune_survival_steps: int = 500  # grace period before an expert may die
    prune_min_usage: float = 1e-4  # usage share below which expert is dying
    max_new_experts: int = 8  # cap per growth event (clone-top-8 strategy)
    # --- BPB ports (all default-off; enabling changes training math) ---
    # use_byte_conv: causal depthwise conv over byte embeddings in the trunk.
    # lookahead_weight/boundary_weight: auxiliary t+2 CE + UTF-8-boundary BCE.
    # cosine_decay_steps: cosine LR decay horizon (0 = constant LR).
    use_byte_conv: bool = False
    lookahead_weight: float = 0.0
    boundary_weight: float = 0.0
    cosine_decay_steps: int = 0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must split over heads"
        assert 1 <= self.min_depth <= self.max_depth, "need 1 <= min <= max depth"
        assert 0.0 < self.halting_threshold < 1.0, "halting threshold in (0,1)"
        assert 0.0 < self.halt_prior < 1.0, "halt prior in (0,1)"
        assert 1 <= self.top_k <= self.num_experts, "need 1 <= top_k <= experts"
        assert self.num_experts >= 1 and self.max_experts >= self.num_experts
        assert self.min_experts >= self.top_k, "need min_experts >= top_k"
        # Note: min_experts may exceed num_experts (e.g. the default 64 floor
        # on a tiny 8-expert pool). Small pools simply never prune: the
        # offline pruner caps removal at len(pool) - floor <= 0. The floor
        # binds once a pool grows to 64+.
        assert self.paging_method in ("D2R", "R2VR", "D2VR"), "bad paging method"
        assert self.ram_cache >= 1 and self.vram_cache >= 1, "caches need >= 1 slot"
        assert self.dtype in ("bf16", "fp32"), "compute dtype bf16|fp32"
        assert self.state_dtype in ("bf16", "fp32"), "state dtype bf16|fp32"
        assert self.expert_hidden >= 8, "expert hidden dim too small"
        assert self.attention_chunk_size >= 1, "attention chunk size must be positive"
        assert self.capacity_factor > 0, "capacity factor must be positive"
        assert self.moe_balance_weight >= 0, "balance weight must be non-negative"
        assert self.expert_lr > 0, "expert LR must be positive"
        assert self.trunk_lr_mult > 0, "trunk LR multiplier must be positive"
        assert self.router_lr_mult > 0, "router LR multiplier must be positive"
        assert self.router_lr_mult_new >= 0, "router new-mode LR multiplier must be non-negative"
        assert math.isfinite(self.new_routing_bias) and self.new_routing_bias >= 0, \
            "new routing bias must be a finite value >= 0"
        assert self.weight_decay >= 0, "weight decay must be non-negative"
        # Clipping semantics: grad_clip > 0 caps the global grad norm;
        # 0 (or negative) disables clipping entirely.
        assert self.grad_clip >= 0, "grad clip must be non-negative (0 disables)"
        assert 0.0 <= self.beta_m < 1.0, "beta_m in [0, 1)"
        assert 0.0 <= self.beta_v < 1.0, "beta_v in [0, 1)"
        assert self.epsilon > 0, "epsilon must be positive"
        assert self.fp8_tile >= 8, "FP8 tile too small"
        assert self.threads >= 1, "need at least 1 thread"
        assert self.context_length >= 2, "context must fit input + target"
        assert self.rmsnorm_eps > 0, "rmsnorm eps must be positive"
        assert self.grow_every >= 0, "grow_every must be non-negative (0 disables)"
        assert self.prune_survival_steps >= 1, "need a positive prune grace period"
        assert self.prune_min_usage >= 0, "prune usage threshold must be non-negative"
        assert self.max_new_experts >= 1, "need at least 1 new expert per growth"
        assert self.lookahead_weight >= 0, "lookahead weight must be non-negative"
        assert self.boundary_weight >= 0, "boundary weight must be non-negative"
        assert self.cosine_decay_steps >= 0, "cosine horizon must be non-negative (0 disables)"
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

    def router_params(self) -> int:
        """Router projection params: [num_experts, d_model] weight + bias."""
        return self.num_experts * self.d_model + self.num_experts

    def shared_params(self) -> int:
        """Shared trunk params (router excluded; see router_params).

        embeddings + attention + norms + halt + head. The recurrent block
        is counted once (unique params); unrolled logical depth reuses it.
        Matches model.param_counts()["shared_params"] (trunk-only).
        """
        d, v = self.d_model, self.vocab_size
        # embed(v,d) + qkv+o (4*d*d) + 6 norms (n_init,n1,n_attn,n2,n3,n_final)
        # + halt (d+1) + out head (v*d)
        return v * d + 4 * d * d + 6 * d + (d + 1) + v * d

    def total_params(self) -> int:
        """Logical == unique params: shared + router + all experts."""
        return self.shared_params() + self.router_params() + self.num_experts * self.per_expert_params

    def active_params(self) -> int:
        """Params touched per token: shared + router + selected experts.

        Selected experts are min(top_k, num_experts): a pruned pool below
        top_k cannot touch more experts than exist (active <= total).
        Admitted (capacity) experts are <= selected; padding/pondering/
        paging/native never change this count.
        """
        return self.shared_params() + self.router_params() + min(self.top_k, self.num_experts) * self.per_expert_params

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SmaulBrainConfig":
        """Rebuild from a serialized dict (checkpoint config.json compatible).

        Compatibility contract:
        - Omitted fields fall back to dataclass defaults (backward compatible).
        - Explicit nulls are rejected (omitted != null; null is never valid).
        - Unknown fields (incl. ``schema_version`` stamped by storage.py,
          whose major is gated there) are tolerated and dropped (forward
          compatible). Use to_dict() for an exact round-trip.
        - ``active_experts`` must equal ``top_k`` when both are present;
          mismatched persisted values raise instead of silently syncing.
        - Impossible floors (``min_experts < top_k``) raise via __post_init__
          instead of being silently clamped (no migration path: old
          checkpoints carrying such floors are corrupt, not loadable).
        """
        if not isinstance(d, dict):
            raise ValueError(f"config payload must be a dict, got {type(d).__name__}")
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in d.items() if k in known}
        nulls = [k for k, v in values.items() if v is None]
        if nulls:
            raise ValueError(
                f"config fields must not be null (omit for defaults): {sorted(nulls)}"
            )
        if "active_experts" in values and "top_k" in values:
            if int(values["active_experts"]) != int(values["top_k"]):
                raise ValueError(
                    f"incompatible active_experts={values['active_experts']!r} "
                    f"!= top_k={values['top_k']!r}; refusing to guess"
                )
        return cls(**values)

    def describe_counts(self) -> dict:
        """Architecture param breakdown with unambiguous count categories.

        - logical_params == unique_params == total_params here: every expert
          is distinct and the shared recurrent block is counted once
          (unrolled depth reuses it, so no depth multiplier).
        - active_params: shared + router + min(top_k, expert_count)
          selected experts per token (upper bound; capacity-admitted <=
          selected). Throughput timing (bytes/s) is not derived here;
          countable throughput facts are 1 token == 1 byte (byte vocab)
          and valid-byte (non-PAD) normalization — see train/bench.
        - resident (RAM/VRAM dequantized transients) and on-disk
          (stored_expert_bytes, FP8 reality) need a live model; see
          model.param_counts(). They are reported as None here to keep the
          categories distinct instead of conflating them.
        """
        return {
            "shared_params": self.shared_params(),
            "router_params": self.router_params(),
            "per_expert_params": self.per_expert_params,
            "expert_count": self.num_experts,
            "expert_params_total": self.num_experts * self.per_expert_params,
            "total_params": self.total_params(),
            "logical_params": self.total_params(),
            "unique_params": self.total_params(),
            "active_params": self.active_params(),
            "resident_ram_params": None,
            "resident_vram_params": None,
            "stored_expert_bytes": None,
        }
