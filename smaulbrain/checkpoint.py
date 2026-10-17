"""Atomic, checksummed PyTorch checkpoints with fallback recovery."""
import hashlib, json, os, shutil, tempfile
import torch
from . import vocab as _vocab
from smaulbrain import CHECKPOINT_VERSION
LEGACY_VERSION="smaulbrain-v2:1"


def _sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def blank_scheduler_state(growth_every_steps=10000,growth_loss_nats=0.5,loss_beta=0.98):
    interval=int(growth_every_steps)
    return {"global_step":0,"growth_every_steps":interval,"next_periodic_step":interval if interval>0 else None,
            "loss_threshold":growth_loss_nats,"loss_beta":float(loss_beta),"smooth_loss":None,"loss_armed":True,
            "event_count":0,"last_growth_step":None,"history":[],"pending":None}


def build_manifest(preset_name,arch,extra,sched,data_cursor,rng_state,n_experts_current,expert_registry,opt_meta):
    return {"version":CHECKPOINT_VERSION,"preset":preset_name,"preset_expert_count_init":arch.get("experts_init"),
            "n_experts_current":n_experts_current,"expert_registry":expert_registry,"vocab":_vocab.vocab_meta(),
            "arch":arch,"config":extra,"scheduler":sched,"data_cursor":int(data_cursor),"rng_state":rng_state,
            "optimizer":opt_meta,"files":{}}


def _cpu_tree(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu().contiguous()
    if isinstance(obj,dict):return {k:_cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return type(obj)(_cpu_tree(v) for v in obj)
    return obj


def _write_tree(target_dir,manifest,params,opt_state):
    parent=os.path.dirname(os.path.abspath(target_dir)); os.makedirs(parent,exist_ok=True)
    tmp=tempfile.mkdtemp(prefix=".ckpt.",dir=parent)
    try:
        pp=os.path.join(tmp,"params.pt"); op=os.path.join(tmp,"optim.pt")
        torch.save(_cpu_tree(params),pp); torch.save(_cpu_tree(opt_state),op)
        man=dict(manifest); man["version"]=CHECKPOINT_VERSION
        man["files"]={"params.pt":_sha256(pp),"optim.pt":_sha256(op)}
        with open(os.path.join(tmp,"manifest.json"),"w") as f:json.dump(man,f,indent=2,default=_json_default)
        _publish_dir(tmp,target_dir); return target_dir
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True); raise


def _publish_dir(tmp,target_dir):
    target_dir=os.path.normpath(target_dir)
    if not os.path.exists(target_dir):os.replace(tmp,target_dir); return
    bak=target_dir+".bak"
    if os.path.exists(bak):shutil.rmtree(bak)
    os.rename(target_dir,bak)
    try:os.rename(tmp,target_dir)
    except BaseException:os.rename(bak,target_dir); raise
    shutil.rmtree(bak,ignore_errors=True)


def _resolve_checkpoint_dir(ckpt_dir):
    if os.path.exists(os.path.join(ckpt_dir,"manifest.json")):return ckpt_dir
    bak=os.path.normpath(ckpt_dir)+".bak"
    return bak if os.path.exists(os.path.join(bak,"manifest.json")) else ckpt_dir


def _json_default(o):
    if isinstance(o,torch.Tensor):return o.detach().cpu().tolist()
    if hasattr(o,"item"):return o.item()
    if hasattr(o,"tolist"):return o.tolist()
    raise TypeError(f"unserializable {type(o)}")


def save_checkpoint(target_dir,manifest,params,opt_state):
    hist=manifest.get("scheduler",{}).get("history",[])
    if any(not isinstance(c.get("id"),int) for c in hist):raise ValueError("committed history entries require integer ids")
    ids=[c["id"] for c in hist]
    if len(set(ids))!=len(ids):raise ValueError("duplicate committed event ids")
    pending=manifest.get("scheduler",{}).get("pending")
    if pending is not None and pending.get("id") in ids:raise ValueError("pending event id already committed")
    _check_consistency(manifest,params)
    return _write_tree(target_dir,manifest,params,opt_state)


def _check_consistency(manifest,params):
    n=manifest.get("n_experts_current"); reg=manifest.get("expert_registry",[])
    if n is not None and reg and len(reg)!=n:raise ValueError("expert registry length disagrees with n_experts_current")
    if n is not None and manifest.get("preset","").startswith("moe"):
        eids=sorted({int(k.split(".")[1]) for k in params if k.startswith("expert.") and k.endswith(".Win")})
        if eids!=list(range(len(eids))):raise ValueError("expert params are not a dense 0..E-1 registry")
        if len(eids)!=n:raise ValueError("param expert count disagrees with manifest")


def load_checkpoint(ckpt_dir):
    ckpt_dir=_resolve_checkpoint_dir(ckpt_dir); mp=os.path.join(ckpt_dir,"manifest.json")
    if not os.path.exists(mp):raise FileNotFoundError(f"no manifest at {mp}")
    with open(mp) as f:man=json.load(f)
    version=man.get("version")
    if version not in (CHECKPOINT_VERSION,LEGACY_VERSION):raise ValueError(f"incompatible checkpoint version {version!r}")
    if version==CHECKPOINT_VERSION:
        names=("params.pt","optim.pt")
        for name in names:
            path=os.path.join(ckpt_dir,name)
            if not os.path.exists(path):raise FileNotFoundError(f"checkpoint missing {name} (incomplete write?)")
            if man.get("files",{}).get(name) not in (None,_sha256(path)):raise ValueError(f"integrity mismatch for {name}")
        params=torch.load(os.path.join(ckpt_dir,"params.pt"),map_location="cpu",weights_only=True)
        opt_state=torch.load(os.path.join(ckpt_dir,"optim.pt"),map_location="cpu",weights_only=True)
    else:
        import numpy as np
        for name in ("params.npz","optim.npz"):
            path=os.path.join(ckpt_dir,name)
            if not os.path.exists(path):raise FileNotFoundError(f"checkpoint missing {name}")
            if man.get("files",{}).get(name) not in (None,_sha256(path)):raise ValueError(f"integrity mismatch for {name}")
        with np.load(os.path.join(ckpt_dir,"params.npz")) as z:params={k:torch.from_numpy(z[k].copy()) for k in z.files}
        with np.load(os.path.join(ckpt_dir,"optim.npz")) as z:
            flat={k:torch.from_numpy(z[k].copy()) for k in z.files}
        opt_state={}
        for k,v in flat.items():name,_,sub=k.partition("::");opt_state.setdefault(name,{})[sub]=v
    params={k:(v if isinstance(v,torch.Tensor) else torch.as_tensor(v)) for k,v in params.items()}
    _check_consistency(man,params)
    return {"manifest":man,"params":params,"opt_state":opt_state,"dir":ckpt_dir}
