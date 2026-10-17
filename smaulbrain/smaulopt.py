"""PyTorch implementation of SmaulOPT's factored second-moment optimizer."""
import math

import torch
from . import _native_opt


class SmaulOPT:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.0):
        self.lr=float(lr)
        self.beta1=float(beta1)
        self.beta2=float(beta2)
        self.eps=float(eps)
        self.weight_decay=float(weight_decay)
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and > 0")
        if not math.isfinite(self.beta1) or not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be finite and in [0, 1)")
        if not math.isfinite(self.beta2) or not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be finite and in [0, 1)")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and > 0")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and >= 0")
        self.state={}
        self.n_updates=0

    def _init_state(self,name,w):
        if w.ndim==2:
            self.state[name]={"m":torch.zeros_like(w,device=w.device),"R":torch.zeros(w.shape[0],device=w.device),
                              "C":torch.zeros(w.shape[1],device=w.device),"factored":True}
        else:
            self.state[name]={"m":torch.zeros_like(w,device=w.device),"v":torch.zeros_like(w,device=w.device),"factored":False}

    def register(self,params):
        for k,w in params.items():
            if k not in self.state:self._init_state(k,w)

    def drop(self,names):
        for k in names:self.state.pop(k,None)

    def state_memory_elements(self):
        factored=full=0
        for s in self.state.values():
            if s["factored"]:factored+=s["m"].numel()+s["R"].numel()+s["C"].numel()
            else:full+=2*s["m"].numel()
        return {"factored_elements":factored,"full_elements":full}

    def step(self,params,grads,collect_norms=True):
        norms={}; b1=self.beta1; b2=self.beta2; eps=self.eps
        with torch.no_grad():
            for name,w in params.items():
                g=grads.get(name)
                if g is None:continue
                g=g.detach().to(device=w.device,dtype=torch.float32).contiguous()
                if name not in self.state:self._init_state(name,w)
                st=self.state[name]
                if st["factored"]:
                    native=_native_opt.factored_step(self,w,g,st,collect_norms)
                    if native is not None:
                        if collect_norms:norms[name]=native
                        continue
                if self.weight_decay:w.mul_(1.0-self.lr*self.weight_decay)
                st["m"].mul_(b1).add_(g,alpha=1.0-b1)
                g2=g.square()
                if st["factored"]:
                    st["R"].mul_(b2).add_(g2.mean(dim=1),alpha=1.0-b2)
                    st["C"].mul_(b2).add_(g2.mean(dim=0),alpha=1.0-b2)
                    denom_r=st["R"].mean().clamp_min(1e-30)
                    vhat=(st["R"][:,None]*st["C"][None,:]/denom_r).clamp_min(0)
                else:
                    st["v"].mul_(b2).add_(g2,alpha=1.0-b2); vhat=st["v"].clamp_min(0)
                upd=st["m"]/(vhat.sqrt()+eps)
                if collect_norms:norms[name]=float(upd.square().mean().sqrt().item())
                w.add_(upd,alpha=-self.lr)
        self.n_updates+=1
        return norms

    def state_dict(self):
        return {k:{sk:(sv.clone() if isinstance(sv,torch.Tensor) else sv) for sk,sv in s.items() if sk!="factored"} for k,s in self.state.items()}

    def load_state_dict(self,sd,factored_flags=None):
        self.state={}
        for k,s in sd.items():
            entry={sk:(torch.as_tensor(sv).clone() if not isinstance(sv,torch.Tensor) else sv.clone()) for sk,sv in s.items()}
            entry["factored"]=bool(factored_flags.get(k,"R" in entry) if factored_flags else "R" in entry)
            self.state[k]=entry
