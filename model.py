"""SmaulBRAIN model: recurrent byte-level LM with adaptive depth.

Pipeline per forward pass:
  byte ids -> embedding -> RMSNorm(n_init) -> shared recurrent state
  -> [SharedRecurrentBlock x D] (RMSNorm -> linear attention -> RMSNorm ->
     residual -> sparse top-k routing -> dynamic MoE -> weighted combination
     -> RMSNorm -> residual -> halt head) -> final RMSNorm -> byte head

Depth D is adaptive in [min_depth, max_depth]: per-token halting
probabilities (PonderNet-style, as verified in mini-AGI's RecurCoder) decide
early exit; the ponder KL against a geometric prior regularizes depth.
Training minimizes sum_n p_n * CE(logits_n) + ponder_beta * KL + MoE balance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SmaulBrainConfig
from bytes import PAD_ID
from experts import ExpertPool, make_expert
from linear_attention import LinearAttnState
from paging import ExpertPager
from recurrent import SharedRecurrentBlock
from rmsnorm import RMSNorm
from routing import SparseRouter


class SmaulBrainModel(nn.Module):
    """Shared trunk + router + dynamic expert pool + pager."""

    def __init__(self, config: SmaulBrainConfig) -> None:
        super().__init__()
        self.cfg = config
        d = config.d_model
        self.embed = nn.Embedding(config.vocab_size, d)
        self.n_init = RMSNorm(d, config.rmsnorm_eps)
        self.block = SharedRecurrentBlock(d, config.n_heads, config.rmsnorm_eps)
        self.n_final = RMSNorm(d, config.rmsnorm_eps)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.router = SparseRouter(d, config.num_experts, config.top_k,
                                   config.capacity_factor)
        self.pool = ExpertPool()
        for _ in range(config.num_experts):
            self.pool.add(make_expert(self.pool.fresh_id(), d, config.expert_hidden,
                                      fp8_tile=config.fp8_tile))
        assert len(self.pool) == config.num_experts
        self.pager = ExpertPager(self.pool, mode=config.paging_method,
                                 ram_cache=config.ram_cache,
                                 vram_cache=config.vram_cache,
                                 compute_dtype=(torch.bfloat16 if config.dtype == "bf16"
                                                else torch.float32))
        if config.paging_method == "R2VR":
            self.pager.warm_ram()
        # Trunk linears/embeddings run in the compute dtype (precision policy:
        # BF16 activations; RMSNorm weights intentionally stay FP32).
        if config.dtype == "bf16":
            for mod in (self.embed, self.block.qkv, self.block.o_proj,
                        self.block.halt, self.head, self.router.proj):
                mod.to(torch.bfloat16)
        # Active expert leaves (training): eid -> list of leaf weight dicts
        # (one entry per depth-step activation; grads are summed at opt time).
        self._leaves: dict[str, list[dict[str, torch.Tensor]]] = {}
        self._last_step: int = 0

    # -- MoE wiring --
    def _train_provider(self, expert_id: str) -> dict[str, torch.Tensor]:
        base = self.pager.provider(expert_id)
        leaves = {k: v.detach().to(base[k].dtype).requires_grad_(True)
                  for k, v in base.items()}
        self._leaves.setdefault(expert_id, []).append(leaves)
        return leaves

    def _eval_provider(self, expert_id: str) -> dict[str, torch.Tensor]:
        return self.pager.provider(expert_id)

    def _moe_fn(self, train: bool, step: int, keep: torch.Tensor | None = None):
        def fn(x: torch.Tensor):
            # Capacity binds only in training: it is batch-size relative,
            # so enforcing it at inference would make chunked/streaming
            # results differ from a full pass over the same tokens.
            plan = self.router.route(x, enforce_capacity=train)
            prov = self._train_provider if train else self._eval_provider
            y = self.pool.forward(x, plan.top_ids, plan.top_weights,
                                  plan.dropped, prov, step=step)
            # Balance over scored positions only: padding must not dilute it.
            aux = self.router.balance_loss(plan.probs, keep=keep)
            return y, aux, plan.top_ids
        return fn

    def take_expert_grads(self) -> dict[str, dict[str, torch.Tensor]]:
        """Sum leaf grads per expert across depth-step copies; clear the buffer.

        Returns eid -> {wname: summed grad or None}. Call after loss.backward().
        """
        agg: dict[str, dict[str, torch.Tensor]] = {}
        for eid, copies in self._leaves.items():
            summed: dict[str, torch.Tensor] = {}
            for leaves in copies:
                for n, leaf in leaves.items():
                    if leaf.grad is None:
                        continue
                    g = leaf.grad.detach().float()
                    summed[n] = g if n not in summed else summed[n] + g
            agg[eid] = summed
        self._leaves = {}
        return agg

    # -- adaptive depth loop --
    def _depth_loop(
        self,
        h: torch.Tensor,
        attn_states: list[LinearAttnState],
        train: bool,
        step: int,
        keep: torch.Tensor | None = None,
    ):
        """Run the shared block; return per-step (h, halt_prob, aux) + depths.

        All max_depth applications always execute: per-depth attention states
        must each observe every token, or streaming inference would resume
        from incomplete states. Adaptivity lives in the per-token halting
        depths (which representation is read out), not in skipped compute.
        ``keep`` (bool [B*T], training only) restricts the MoE balance loss
        to scored positions so padding cannot dilute it.
        """
        B, T, _D = h.shape
        hs: list[torch.Tensor] = []
        lams: list[torch.Tensor] = []
        aux_total = torch.zeros((), device=h.device)
        depths = torch.full((B, T), self.cfg.max_depth, dtype=torch.long, device=h.device)
        cum = torch.zeros(B, T, device=h.device)
        halted = torch.zeros(B, T, dtype=torch.bool, device=h.device)
        n_executed = 0
        for depth in range(self.cfg.max_depth):
            h, attn_states[depth], halt_logit, aux = self.block(
                h, attn_states[depth], self._moe_fn(train, step, keep=keep),
                chunk_size=self.cfg.attention_chunk_size,
            )
            aux_total = aux_total + aux  # keep router grad graph (training)
            lam = torch.sigmoid(halt_logit.float())  # [B, T]
            hs.append(h)
            lams.append(lam)
            n_executed = depth + 1
            if train:
                continue  # training always runs max_depth for ponder weighting
            if n_executed >= self.cfg.min_depth:
                cum = cum + (1.0 - cum) * lam.detach()
                newly = (~halted) & (cum >= self.cfg.halting_threshold)
                depths[newly] = n_executed
                halted = halted | (cum >= self.cfg.halting_threshold)
        return hs, lams, aux_total, depths, n_executed

    def _ponder(self, lams: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Halting distribution p_n, mean geometric-prior KL, per-position KL."""
        B, T = lams[0].shape
        dev = lams[0].device
        p_pr = self.cfg.halt_prior
        probs: list[torch.Tensor] = []
        remaining = torch.ones(B, T, device=dev)
        kl_pos = torch.zeros(B, T, device=dev)
        for n, lam in enumerate(lams):
            if n == len(lams) - 1:
                p = remaining  # force-stop: all remaining mass halts here
            else:
                p = lam * remaining
                remaining = remaining * (1.0 - lam)
            probs.append(p)
            geom = p_pr * ((1.0 - p_pr) ** n)
            kl_pos = kl_pos + p * (torch.log(p.clamp_min(1e-9)) - math.log(max(geom, 1e-12)))
        P = torch.stack(probs, dim=0)  # [D, B, T]
        return P, kl_pos.mean(), kl_pos

    # -- forward --
    def _check_ids(self, ids: torch.Tensor, what: str) -> None:
        """Fail loudly on out-of-vocabulary ids (never a cryptic IndexError)."""
        if ids.numel() == 0:
            return
        lo, hi = int(ids.min().item()), int(ids.max().item())
        if not 0 <= lo or hi >= self.cfg.vocab_size:
            raise ValueError(
                f"{what} has ids outside [0, {self.cfg.vocab_size}) "
                f"(min={lo}, max={hi})"
            )

    def forward(
        self, ids: torch.Tensor, targets: torch.Tensor | None = None, step: int = 0
    ) -> dict:
        """Training forward (grad enabled by caller). Returns logits/loss/stats."""
        self._check_ids(ids, "ids")
        if targets is not None:
            self._check_ids(targets, "targets")
        self._last_step = step
        self._leaves = {}
        compute = torch.bfloat16 if self.cfg.dtype == "bf16" else torch.float32
        B, T = ids.shape
        # Scored positions: padding is not data and is excluded from the MoE
        # balance loss as well as the CE/KL/accuracy below.
        valid = (targets != PAD_ID) if targets is not None else None
        h = self.n_init(self.embed(ids).to(compute))
        attn_states = [
            LinearAttnState.zeros(
                B, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads,
                device=h.device
            )
            for _ in range(self.cfg.max_depth)
        ]
        hs, lams, aux, depths, n_executed = self._depth_loop(
            h, attn_states, train=True, step=step,
            keep=valid.reshape(-1) if targets is not None else None,
        )
        P, _kl_mean, kl_pos = self._ponder([l.float() for l in lams])
        if targets is None:
            # Same ponder-mixed readout as the scored path (no targets to
            # mask; P sums to 1 per position regardless).
            mixed = sum(P[n].detach() * self.head(self.n_final(hs[n])).float()
                        for n in range(len(hs)))
            return {"logits": mixed, "depths": depths, "n_executed": n_executed}
        # Ponder-weighted CE: each step's logits score against halting mass.
        # Padding (PAD_ID) is not data: it is excluded from the loss via
        # ignore_index and every mean below normalizes over valid targets
        # only, so trailing-pad length cannot dilute (or NaN) the loss.
        assert valid is not None
        n_valid = int(valid.sum().item())
        ce_steps = []
        mixed = 0.0  # ponder-mixed readout: the predictor the loss optimizes
        for n, h_n in enumerate(hs):
            logits_n = self.head(self.n_final(h_n)).float()
            mixed = mixed + P[n].unsqueeze(-1).detach() * logits_n.detach()
            ce = F.cross_entropy(logits_n.reshape(-1, self.cfg.vocab_size),
                                 targets.reshape(-1), reduction="none",
                                 ignore_index=PAD_ID).reshape(B, T)
            ce_steps.append(ce)
        CE = torch.stack(ce_steps, dim=0)  # [D, B, T]
        # Mask before summing: the same valid values in the same order sum
        # bit-identically regardless of how many pads surround them.
        pondered = (P * CE).sum(dim=0)  # [B, T] per-position pondered CE
        nll = pondered[valid].sum() / max(1, n_valid)
        kl_valid = kl_pos[valid].sum() / max(1, n_valid)
        balance = aux / n_executed
        loss = nll + self.cfg.ponder_beta * kl_valid + self.cfg.moe_balance_weight * balance
        with torch.no_grad():
            # Accuracy uses the ponder-mixed readout (same predictor the
            # loss scores), not the max-depth representation inference
            # would not necessarily read out.
            pred = mixed.argmax(-1)
            acc = ((pred == targets) & valid).float().sum().item() / max(1, n_valid)
            steps = torch.arange(1, len(hs) + 1, device=P.device).view(-1, 1, 1)
            mean_depth = float((P * steps).sum().item() / (B * T))
        return {
            # The model's prediction is the ponder-mixed readout, matching
            # the loss and the accuracy above (cached: no second head pass).
            "logits": mixed,
            "loss": loss, "nll": nll.detach(), "ponder_kl": kl_valid.detach(),
            "balance": balance.detach() if torch.is_tensor(balance) else balance,
            "acc": acc, "depths": depths, "mean_depth": mean_depth,
            "n_executed": n_executed,
        }

    @torch.no_grad()
    def forward_infer(self, ids: torch.Tensor, step: int = 0) -> dict:
        """Inference forward: fixed-state recurrent pass with token-depth selection."""
        self._check_ids(ids, "ids")
        compute = torch.bfloat16 if self.cfg.dtype == "bf16" else torch.float32
        B, T = ids.shape
        h = self.n_init(self.embed(ids).to(compute))
        attn_states = [
            LinearAttnState.zeros(
                B, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads,
                device=h.device
            )
            for _ in range(self.cfg.max_depth)
        ]
        hs, lams, _aux, depths, n_executed = self._depth_loop(
            h, attn_states, train=False, step=step
        )
        selected_h = self._select_depth_hidden(hs, depths)
        logits = self.head(self.n_final(selected_h)).float()
        return {"logits": logits, "depths": depths, "n_executed": n_executed,
                "halt_probs": [l.detach() for l in lams]}

    @staticmethod
    def _select_depth_hidden(
        hs: list[torch.Tensor], depths: torch.Tensor
    ) -> torch.Tensor:
        """Select each token's halting-depth representation."""
        stack = torch.stack(hs, dim=0)  # [D, B, T, H]
        idx = (depths - 1).clamp(0, len(hs) - 1)
        width = stack.shape[-1]
        gather_idx = idx.unsqueeze(0).unsqueeze(-1).expand(1, *idx.shape, width)
        return stack.gather(0, gather_idx).squeeze(0)

    def new_infer_state(self, batch: int = 1) -> list[LinearAttnState]:
        """Create the fixed-size per-depth attention state for streaming."""
        dev = self.embed.weight.device
        return [
            LinearAttnState.zeros(
                batch, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads,
                device=dev
            )
            for _ in range(self.cfg.max_depth)
        ]

    @torch.no_grad()
    def forward_infer_stateful(
        self,
        ids: torch.Tensor,
        attn_states: list[LinearAttnState] | None = None,
        step: int = 0,
    ) -> tuple[dict, list[LinearAttnState]]:
        """Process a prompt/chunk and return its updated fixed-size recurrent states."""
        self._check_ids(ids, "ids")
        compute = torch.bfloat16 if self.cfg.dtype == "bf16" else torch.float32
        B, _T = ids.shape
        h = self.n_init(self.embed(ids).to(compute))
        if attn_states is None:
            attn_states = self.new_infer_state(B)
        if len(attn_states) != self.cfg.max_depth:
            raise ValueError("attention state depth does not match model max_depth")
        for st in attn_states:
            if not st.matches(B, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads):
                raise ValueError(
                    "attention state width does not match model "
                    f"(expected B={B} H={self.cfg.n_heads} "
                    f"Dh={self.cfg.d_model // self.cfg.n_heads})"
                )
        hs, lams, _aux, depths, n_executed = self._depth_loop(
            h, attn_states, train=False, step=step
        )
        selected_h = self._select_depth_hidden(hs, depths)
        logits = self.head(self.n_final(selected_h)).float()
        return {
            "logits": logits,
            "depths": depths,
            "n_executed": n_executed,
            "halt_probs": [l.detach() for l in lams],
        }, attn_states

    @torch.no_grad()
    def forward_infer_step(
        self,
        ids: torch.Tensor,
        attn_states: list[LinearAttnState],
        step: int = 0,
    ) -> tuple[dict, list[LinearAttnState]]:
        """Process one or more new tokens using persistent per-depth state."""
        if ids.ndim == 1:
            ids = ids.unsqueeze(1)
        if ids.ndim != 2 or ids.shape[1] != 1:
            raise ValueError("forward_infer_step expects [B] or [B, 1]")
        return self.forward_infer_stateful(ids, attn_states=attn_states, step=step)

    # -- parameter accounting (dynamic topology) --
    def param_counts(self) -> dict:
        """Logical params (architecture) plus stored expert bytes (FP8 reality).

        Logical counts size the model; ``stored_expert_bytes`` sizes RAM and
        checkpoints. Resident cache counts are dequantized compute elements
        (transients), not parameters — hence reported separately.
        """
        per_expert = self.pool.experts[self.pool.order[0]].param_count if len(self.pool) else 0
        trunk_n = sum(p.nelement() for _, p in self._trunk_params())
        router_n = sum(p.nelement() for _, p in self._router_params())
        stored = sum(rec.storage_bytes() for rec in self.pool.experts.values())
        return {
            "shared_params": trunk_n,
            "router_params": router_n,
            "per_expert_params": per_expert,
            "expert_count": len(self.pool),
            "expert_params_total": per_expert * len(self.pool),
            "stored_expert_bytes": stored,
            "total_params": trunk_n + router_n + per_expert * len(self.pool),
            "active_params": trunk_n + router_n + per_expert * self.cfg.top_k,
            "resident_ram_params": self._resident_params(self.pager.ram),
            "resident_vram_params": self._resident_params(self.pager.vram),
        }

    @staticmethod
    def _resident_params(cache) -> int:
        return sum(v.nelement() for w in cache.values() for v in w.values())

    def _trunk_params(self) -> list[tuple[str, torch.Tensor]]:
        out = []
        for mod, prefix in ((self.embed, "embed"), (self.block, "block"),
                            (self.n_init, "n_init"), (self.n_final, "n_final"),
                            (self.head, "head")):
            for n, p in mod.named_parameters():
                out.append((f"{prefix}.{n}", p))
        return out

    def _router_params(self) -> list[tuple[str, torch.Tensor]]:
        return [(f"router.{n}", p) for n, p in self.router.named_parameters()]
