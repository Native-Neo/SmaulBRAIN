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

from .config import SmaulBrainConfig
from .experts import ExpertPool, make_expert
from .linear_attention import LinearAttnState
from .paging import ExpertPager
from .recurrent import SharedRecurrentBlock
from .rmsnorm import RMSNorm
from .routing import SparseRouter


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
        # Active expert leaves (training): eid -> dequantized leaf weights w/ grad.
        self._leaves: dict[str, dict[str, torch.Tensor]] = {}
        self._last_step: int = 0

    # -- MoE wiring --
    def _train_provider(self, expert_id: str) -> dict[str, torch.Tensor]:
        base = self.pager.provider(expert_id)
        leaves = {k: v.detach().to(base[k].dtype).requires_grad_(True)
                  for k, v in base.items()}
        self._leaves[expert_id] = leaves
        return leaves

    def _eval_provider(self, expert_id: str) -> dict[str, torch.Tensor]:
        return self.pager.provider(expert_id)

    def _moe_fn(self, train: bool, step: int):
        def fn(x: torch.Tensor):
            plan = self.router.route(x)
            prov = self._train_provider if train else self._eval_provider
            y = self.pool.forward(x, plan.top_ids, plan.top_weights,
                                  plan.dropped, prov, step=step)
            aux = self.router.balance_loss(plan.probs)
            return y, aux, plan.top_ids
        return fn

    # -- adaptive depth loop --
    def _depth_loop(self, h: torch.Tensor, attn: LinearAttnState, train: bool, step: int):
        """Run the shared block; return per-step (h, halt_prob, aux) + depths."""
        B, T, _D = h.shape
        hs: list[torch.Tensor] = []
        lams: list[torch.Tensor] = []
        aux_total = torch.zeros((), device=h.device)
        depths = torch.full((B, T), self.cfg.max_depth, dtype=torch.long)
        cum = torch.zeros(B, T)
        halted = torch.zeros(B, T, dtype=torch.bool)
        n_executed = 0
        for depth in range(self.cfg.max_depth):
            h, attn, halt_logit, aux = self.block(h, attn, self._moe_fn(train, step))
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
                if bool(halted.all()):
                    break
        return hs, lams, aux_total, depths, n_executed

    def _ponder(self, lams: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Halting distribution p_n and geometric-prior KL (FP32 stats)."""
        B, T = lams[0].shape
        device = lams[0].device
        p_pr = self.cfg.halt_prior
        probs: list[torch.Tensor] = []
        remaining = torch.ones(B, T)
        kl = torch.zeros(())
        for n, lam in enumerate(lams):
            if n == len(lams) - 1:
                p = remaining  # force-stop: all remaining mass halts here
            else:
                p = lam * remaining
                remaining = remaining * (1.0 - lam)
            probs.append(p)
            if n < len(lams) - 1 or True:
                geom = p_pr * ((1.0 - p_pr) ** n)
                kl = kl + (p * (torch.log(p.clamp_min(1e-9)) - math.log(max(geom, 1e-12)))).sum()
        P = torch.stack(probs, dim=0)  # [D, B, T]
        return P, kl / (B * T)

    # -- forward --
    def forward(
        self, ids: torch.Tensor, targets: torch.Tensor | None = None, step: int = 0
    ) -> dict:
        """Training forward (grad enabled by caller). Returns logits/loss/stats."""
        self._last_step = step
        self._leaves = {}
        compute = torch.bfloat16 if self.cfg.dtype == "bf16" else torch.float32
        B, T = ids.shape
        h = self.n_init(self.embed(ids).to(compute))
        attn = LinearAttnState.zeros(B, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads)
        attn.S = attn.S.to(h.device)
        attn.z = attn.z.to(h.device)
        hs, lams, aux, depths, n_executed = self._depth_loop(h, attn, train=True, step=step)
        P, kl = self._ponder([l.float() for l in lams])
        if targets is None:
            logits = self.head(self.n_final(hs[-1])).float()
            return {"logits": logits, "depths": depths, "n_executed": n_executed}
        # Ponder-weighted CE: each step's logits score against halting mass.
        ce_steps = []
        for h_n in hs:
            logits_n = self.head(self.n_final(h_n)).float()
            ce = F.cross_entropy(logits_n.reshape(-1, self.cfg.vocab_size),
                                 targets.reshape(-1), reduction="none").reshape(B, T)
            ce_steps.append(ce)
        CE = torch.stack(ce_steps, dim=0)  # [D, B, T]
        nll = (P * CE).sum(dim=0).mean()
        balance = aux / n_executed
        loss = nll + self.cfg.ponder_beta * kl + self.cfg.moe_balance_weight * balance
        with torch.no_grad():
            pred = self.head(self.n_final(hs[-1])).float().argmax(-1)
            acc = (pred == targets).float().mean().item()
            steps = torch.arange(1, len(hs) + 1, device=P.device).view(-1, 1, 1)
            mean_depth = float((P * steps).sum().item() / (B * T))
        return {
            "logits": self.head(self.n_final(hs[-1])).float(),
            "loss": loss, "nll": nll.detach(), "ponder_kl": kl.detach(),
            "balance": balance.detach() if torch.is_tensor(balance) else balance,
            "acc": acc, "depths": depths, "mean_depth": mean_depth,
            "n_executed": n_executed,
        }

    @torch.no_grad()
    def forward_infer(self, ids: torch.Tensor, step: int = 0) -> dict:
        """Inference forward: early halting, no gradients."""
        compute = torch.bfloat16 if self.cfg.dtype == "bf16" else torch.float32
        B, T = ids.shape
        h = self.n_init(self.embed(ids).to(compute))
        attn = LinearAttnState.zeros(B, self.cfg.n_heads, self.cfg.d_model // self.cfg.n_heads)
        hs, lams, _aux, depths, n_executed = self._depth_loop(h, attn, train=False, step=step)
        logits = self.head(self.n_final(hs[-1])).float()
        return {"logits": logits, "depths": depths, "n_executed": n_executed,
                "halt_probs": [l.detach() for l in lams]}

    # -- parameter accounting (dynamic topology) --
    def param_counts(self) -> dict:
        d = self.cfg.d_model
        per_expert = self.pool.experts[self.pool.order[0]].param_count if len(self.pool) else 0
        shared = sum(p.nelement() for _, p in self._trunk_params()) + sum(
            p.nelement() for _, p in self._router_params())
        router_n = sum(p.nelement() for _, p in self._router_params())
        return {
            "shared_params": shared - router_n,
            "router_params": router_n,
            "per_expert_params": per_expert,
            "expert_count": len(self.pool),
            "expert_params_total": per_expert * len(self.pool),
            "total_params": shared + per_expert * len(self.pool),
            "active_params": shared + per_expert * self.cfg.top_k,
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
