#!/usr/bin/env python3
"""Benchmark PyTorch MoE routing against a simple stable-sort reference."""
import argparse,json,time,torch
from smaulbrain.moe import Router


def stable_reference(router,X,top_k,capacity_factor=1.25,disabled=()):
    scores=X@router.W
    if disabled:
        disabled=list(sorted(set(int(i) for i in disabled)))
        if any(i<0 or i>=scores.shape[1] for i in disabled):
            raise ValueError("disabled expert index out of range")
        scores=scores.clone()
        scores[:,disabled]=-torch.inf
    T,E=scores.shape
    idx=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:top_k]
    values=scores.gather(1,idx)
    raw_weights=torch.softmax(values,dim=1)
    cap=max(int((capacity_factor*T*top_k/E)+0.999999),1)
    keep=torch.ones_like(idx,dtype=torch.bool)
    flat_idx=idx.reshape(-1)
    flat_values=values.reshape(-1)
    flat_keep=keep.reshape(-1)
    for eid in torch.unique(flat_idx,sorted=True).tolist():
        positions=torch.nonzero(flat_idx==eid,as_tuple=False).flatten()
        if positions.numel()>cap:
            order=torch.argsort(flat_values[positions],descending=True,stable=True)
            flat_keep[positions[order[cap:]]]=False
    weights=raw_weights*keep
    weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
    load=torch.bincount(flat_idx[flat_keep],minlength=E)
    return idx,weights,load,keep


def median_seconds(fn,repeat):
    samples=[];result=None
    for _ in range(repeat):
        t=time.perf_counter();result=fn();samples.append(time.perf_counter()-t)
    return float(torch.tensor(samples).median().item()),samples,result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tokens",type=int,default=64);ap.add_argument("--experts",type=int,default=64)
    ap.add_argument("--d-model",type=int,default=512);ap.add_argument("--top-k",type=int,default=8);ap.add_argument("--repeat",type=int,default=7)
    ap.add_argument("--seed",type=int,default=0);ap.add_argument("--device",default="cpu");args=ap.parse_args()
    if min(args.tokens,args.experts,args.d_model,args.top_k,args.repeat)<1:ap.error("all sizes must be positive")
    if args.top_k>args.experts:ap.error("--top-k cannot exceed --experts")
    torch.manual_seed(args.seed);device=torch.device(args.device);router=Router(args.d_model,args.experts,device=device)
    X=torch.randn(args.tokens,args.d_model,device=device)
    ref_s,ref_samples,ref=median_seconds(lambda:stable_reference(router,X,args.top_k),args.repeat)
    opt_s,opt_samples,opt=median_seconds(lambda:router.route(X,args.top_k),args.repeat)
    ids,weights,info,cache=opt
    parity=(torch.equal(ids,ref[0]) and torch.allclose(weights,ref[1],atol=1e-6,rtol=1e-6)
            and torch.equal(torch.tensor(info["load"],device=device),ref[2]) and torch.equal(cache[4],ref[3]))
    if not parity:raise SystemExit("optimized route diverged from stable reference")
    print(json.dumps({"shape":{"tokens":args.tokens,"experts":args.experts,"d_model":args.d_model,"top_k":args.top_k},
          "repeat":args.repeat,"stable_reference_median_s":ref_s,"stable_reference_samples_s":ref_samples,
          "optimized_median_s":opt_s,"optimized_samples_s":opt_samples,"speedup":ref_s/max(opt_s,1e-12),"parity":"passed"},indent=2))
if __name__=="__main__":main()
