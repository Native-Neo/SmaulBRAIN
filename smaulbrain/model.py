"""PyTorch SmaulBRAIN: byte-level recurrent linear attention + shared MoE pool."""
from __future__ import annotations
import math
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as _cfg, moe as _moe, ops as _ops, quant as _quant, vocab as _vocab


def build_params(spec, generator: torch.Generator | None = None, n_experts: int = -1, device="cpu"):
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    elif not isinstance(generator, torch.Generator):
        if hasattr(generator, "integers"):
            seed = int(generator.integers(0, 2**63 - 1))
        elif isinstance(generator, int):
            seed = generator
        else:
            raise TypeError("generator must be a torch.Generator, integer seed, or legacy NumPy Generator")
        generator = torch.Generator(device="cpu").manual_seed(seed)
    d = spec.d_model; p = {}; s = 1.0 / math.sqrt(d)
    def rand(shape, scale=s):
        return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(device)
    p["embed"] = rand((spec.vocab_size, d)); p["final_norm"] = torch.ones(d, device=device)
    for i in range(spec.n_layers):
        px = f"L{i}."
        for key in ("Wr", "Wk", "Wv", "Wo"): p[px + "attn." + key] = rand((d, d))
        p[px + "attn.w"] = rand((d,), 0.1); p[px + "attn.u"] = torch.zeros(d, device=device)
        p[px + "n1"] = torch.ones(d, device=device); p[px + "n2"] = torch.ones(d, device=device)
        if spec.kind == "dense":
            h = spec.ffn_hidden
            p[px + "ffn.Wa"] = rand((d, h)); p[px + "ffn.Wb"] = rand((d, h))
            p[px + "ffn.Wc"] = rand((h, d), 1.0 / math.sqrt(h))
    if spec.kind == "moe":
        n = spec.n_experts_init if n_experts < 0 else n_experts
        p["router"] = rand((d, n))
        for eid in range(n):
            p[f"expert.{eid}.Win"] = rand((d, spec.expert_hidden))
            p[f"expert.{eid}.Wout"] = rand((spec.expert_hidden, d), 1.0 / math.sqrt(spec.expert_hidden))
    return p


def count_params_dict(p): return int(sum(v.numel() for v in p.values()))


def expert_ids(p):
    return sorted({int(k.split(".")[1]) for k in p if k.startswith("expert.")})


class PackedCache:
    def __init__(self, max_entries=256): self.max_entries=max_entries; self.d={}; self.reuse_hits=0; self.builds=0
    def get(self, key, W):
        e=self.d.get(key)
        if e is not None and e["ref"]() is W and e["version"] == W._version:
            self.reuse_hits += 1; return e["packed"], e["scale"], e["shape"]
        packed, scale, shape, _ = _quant.pack_ternary_tiled(W)
        if key not in self.d and len(self.d) >= self.max_entries: self.d.pop(next(iter(self.d)))
        self.d[key]={"packed":packed,"scale":scale,"shape":shape,"ref":weakref.ref(W),"version":W._version}
        self.builds += 1; return packed, scale, shape
    def invalidate(self, keys=None):
        if keys is None: self.d.clear()
        else:
            for key in keys: self.d.pop(key, None)


class FP8Cache:
    def __init__(self, max_entries=256): self.max_entries=max_entries; self.d={}; self.builds=0; self.reuse_hits=0
    def get(self,key,W):
        e=self.d.get(key)
        if e is not None and e["ref"]() is W and e["version"] == W._version:
            self.reuse_hits += 1; return e["weight"], e["scale"]
        Wq,scale=_quant.fp8_e4m3_quantize(W.detach())
        if key not in self.d and len(self.d)>=self.max_entries: self.d.pop(next(iter(self.d)))
        self.d[key]={"weight":Wq.detach(),"scale":scale,"ref":weakref.ref(W),"version":W._version}
        self.builds += 1; return Wq.detach(),scale
    def invalidate(self,keys=None):
        if keys is None: self.d.clear()
        else:
            for key in keys:self.d.pop(key,None)


class _QuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, Wq, kind, backward_mode, packed, scale, shape):
        ctx.kind=kind; ctx.backward_mode=backward_mode; ctx.packed=packed; ctx.scale=scale; ctx.shape=shape
        ctx.save_for_backward(X, W, Wq)
        if kind == "fp32": return X @ W
        if kind == "fp8": return X @ Wq
        return _quant.packed_matmat(X, packed, shape, scale)

    @staticmethod
    def backward(ctx, dY):
        X,W,Wq=ctx.saved_tensors; kind=ctx.kind; mode=ctx.backward_mode
        if kind == "fp32": return dY @ W.T, X.T @ dY, None, None, None, None, None, None
        if kind == "fp8":
            if mode == "fp8":
                dYq,_=_quant.fp8_e4m3_quantize(dY); Xq,_=_quant.fp8_e4m3_quantize(X)
                return dYq @ Wq.T, Xq.T @ dYq, None, None, None, None, None, None
            return dY @ Wq.T, X.T @ dY, None, None, None, None, None, None
        dX=_quant.packed_matmat_transpose(dY,ctx.packed,ctx.shape,ctx.scale)
        if mode == "fp8":
            dYq,_=_quant.fp8_e4m3_quantize(dY); Xq,_=_quant.fp8_e4m3_quantize(X); dW=Xq.T @ dYq
        else: dW=X.T @ dY
        return dX,dW,None,None,None,None,None,None


class Model(nn.Module):
    def __init__(self, spec, params: dict, forward_quant="fp32", backward_mode="fp32",
                 paging_mode="R", paging_store=None, loss_smooth_beta=0.99,
                 capacity_factor: float=1.25, device="cpu"):
        super().__init__(); _cfg.validate_quant_combo(forward_quant,backward_mode)
        self.spec=spec; self.device=torch.device(device); self.param_names=[]; self.params=nn.ParameterList()
        self.p=params
        self.forward_quant=forward_quant; self.backward_mode=backward_mode; self.paging_mode=paging_mode
        self.store=paging_store; self.packed=PackedCache(); self.fp8=FP8Cache()
        self.loss_smooth_beta=float(loss_smooth_beta); self.capacity_factor=float(capacity_factor)
        self._last_aux={"overflow_dropped":0,"balance_aux":0.0}; self._paged_leaves={}
        if forward_quant == "ternary": _quant.prepare_ternary_backend()
        self._assert_counts()

    @property
    def p(self): return {k:v for k,v in zip(self.param_names,self.params)}

    @p.setter
    def p(self, values):
        names=[]; vals=nn.ParameterList()
        for k,v in values.items():
            t=v.detach().to(device=self.device,dtype=torch.float32).contiguous()
            vals.append(nn.Parameter(t,requires_grad=True)); names.append(k)
        self.param_names=names; self.params=vals

    def _assert_counts(self):
        p=self.p
        if self.spec.kind=="dense":
            n=count_params_dict(p); exp=_cfg.EXPECTED_DENSE_TOTAL[self.spec.name]
            if n != exp: raise ValueError(f"preset {self.spec.name}: got {n} params, expected {exp}")
        else:
            for eid in expert_ids(p):
                n=p[f"expert.{eid}.Win"].numel()+p[f"expert.{eid}.Wout"].numel()
                exp=_cfg.EXPECTED_EXPERT_PARAMS[self.spec.name]
                if n != exp: raise ValueError(f"preset {self.spec.name} expert {eid}: got {n}, expected {exp}")

    def report(self):
        n=len(expert_ids(self.p)) if self.spec.kind=="moe" else 0
        rep=_cfg.arch_report(self.spec,n); rep.update(forward_quant=self.forward_quant,backward_mode=self.backward_mode,paging_mode=self.paging_mode)
        return rep

    def _linear(self,X,W,key):
        kind=self.forward_quant; Wq=torch.empty(0,device=X.device)
        packed=None; scale=1.0; shape=tuple(W.shape)
        if kind=="fp8": Wq,scale=self.fp8.get(key,W)
        elif kind=="ternary": packed,scale,shape=self.packed.get(key,W)
        return _QuantLinear.apply(X,W,Wq,kind,self.backward_mode,packed,scale,shape)

    def _expert_weights(self,eid):
        if self.store is None or self.store.mode in ("R","R2V","V"):
            return {"Win":self.p[f"expert.{eid}.Win"],"Wout":self.p[f"expert.{eid}.Wout"]}
        loaded=self.store.get(eid)
        if torch.is_grad_enabled():
            out={}
            for suffix in ("Win","Wout"):
                key=(eid,suffix)
                if key not in self._paged_leaves:
                    self._paged_leaves[key]=loaded[suffix].detach().to(self.device).requires_grad_(True)
                out[suffix]=self._paged_leaves[key]
            return out
        return {k:v.to(self.device) for k,v in loaded.items()}

    def _attn(self,X,px,s0):
        p=self.p; R=self._linear(X,p[px+"attn.Wr"],px+"attn.Wr")
        K=self._linear(X,p[px+"attn.Wk"],px+"attn.Wk"); V=self._linear(X,p[px+"attn.Wv"],px+"attn.Wv")
        alpha=torch.exp(-F.softplus(p[px+"attn.w"]))
        state=torch.zeros_like(alpha) if s0 is None else s0.to(device=X.device,dtype=X.dtype)
        kv=K*V; rows=[]
        for t in range(X.shape[0]):
            state=alpha*state+kv[t]; rows.append(state)
        S=torch.stack(rows) if rows else X.new_zeros((0,X.shape[-1]))
        Y=torch.sigmoid(R)*(S+p[px+"attn.u"]*kv)
        return self._linear(Y,p[px+"attn.Wo"],px+"attn.Wo"),state

    def _ffn(self,X,px):
        p=self.p; A=self._linear(X,p[px+"ffn.Wa"],px+"ffn.Wa"); B=self._linear(X,p[px+"ffn.Wb"],px+"ffn.Wb")
        return self._linear(F.silu(A)*B,p[px+"ffn.Wc"],px+"ffn.Wc")

    def _expert(self,X,eid):
        W=self._expert_weights(eid)
        return self._linear(F.gelu(self._linear(X,W["Win"],f"expert.{eid}.Win"),approximate="tanh"),W["Wout"],f"expert.{eid}.Wout")

    def _moe(self,X,disabled=()):
        p=self.p; router=_moe.Router.__new__(_moe.Router); router.W=p["router"]
        idx,weights,info,cache=router.route(X,self.spec.top_k,self.capacity_factor,disabled)
        Y=X.new_zeros(X.shape)
        flat_idx=idx.reshape(-1)
        kept=cache[-1].reshape(-1)
        for eid in torch.unique(flat_idx[kept],sorted=True).tolist():
            # Do not run experts for assignments already rejected by capacity.
            flatpos=torch.nonzero((flat_idx==eid)&kept,as_tuple=False).flatten()
            rows=torch.div(flatpos,self.spec.top_k,rounding_mode="floor"); cols=flatpos%self.spec.top_k
            if self.store is not None:self.store.pin(eid)
            try:
                xb=X[rows]; yb=self._expert(xb,eid); wt=weights[rows,cols].unsqueeze(1)
                Y=Y.index_add(0,rows,wt*yb)
            finally:
                if self.store is not None:self.store.unpin(eid)
        return Y,info

    def _forward_impl(self,tokens,states=None,disabled=()):
        p=self.p; tok=torch.as_tensor(tokens,dtype=torch.long,device=self.device).reshape(-1)
        X=p["embed"][tok]; new_states=[] if states is None else states
        aux={"overflow_dropped":0,"balance_aux":0.0}
        for i in range(self.spec.n_layers):
            px=f"L{i}."; s0=None if states is None else states[i]
            h=X*torch.rsqrt(X.square().mean(-1,keepdim=True)+_ops.EPS)*p[px+"n1"]
            o,s1=self._attn(h,px,s0); X=X+o
            h2=X*torch.rsqrt(X.square().mean(-1,keepdim=True)+_ops.EPS)*p[px+"n2"]
            if self.spec.kind=="dense": f=self._ffn(h2,px)
            else:
                f,info=self._moe(h2,disabled); aux["overflow_dropped"]+=info["overflow_dropped"]
            X=X+f
            if states is None:new_states.append(s1.detach())
            else:states[i]=s1.detach()
        H=X*torch.rsqrt(X.square().mean(-1,keepdim=True)+_ops.EPS)*p["final_norm"]
        logits=H @ p["embed"].T
        self._last_aux=aux
        return logits,(states if states is not None else new_states)

    def forward(self,tokens,states=None,train=False,disabled=()):
        if train:
            return self._forward_impl(tokens,states,disabled)[0], self._last_aux
        with torch.no_grad(): return self._forward_impl(tokens,states,disabled)

    def loss_and_backward(self,tokens):
        tok=torch.as_tensor(tokens,dtype=torch.long,device=self.device).reshape(-1)
        if tok.numel()<2: raise ValueError("training sequence needs at least two tokens")
        self.zero_grad(set_to_none=True); self._paged_leaves={}
        with torch.enable_grad():
            logits,_=self._forward_impl(tok[:-1])
            loss=F.cross_entropy(logits,tok[1:])
            loss.backward()
        p=self.p
        for (eid,suffix),leaf in self._paged_leaves.items():
            target=p[f"expert.{eid}.{suffix}"]
            if leaf.grad is not None:
                if target.grad is None: target.grad=leaf.grad.detach()
                else: target.grad.add_(leaf.grad.detach())
        # Keep the gradient storage itself alive through the returned dict; zero_grad(set_to_none=True) below only detaches it from parameters, so cloning every full-size gradient is unnecessary.
        grads={k:v.grad.detach() for k,v in p.items() if v.grad is not None}
        out=float(loss.detach().item()); aux=dict(self._last_aux); self.zero_grad(set_to_none=True)
        return out,grads,aux

    def init_states(self): return [torch.zeros(self.spec.d_model,dtype=torch.float32,device=self.device) for _ in range(self.spec.n_layers)]
    @staticmethod
    def serialize_states(states): return {str(i):s.detach().cpu().tolist() for i,s in enumerate(states)}
    @staticmethod
    def restore_states(blob): return [torch.as_tensor(blob[str(i)],dtype=torch.float32) for i in range(len(blob))]
    def step_token(self,tok,states):
        logits,_=self.forward([tok],states=states,train=False); return logits[0]
