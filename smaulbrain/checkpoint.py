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
    if not os.path.exists(target_dir):
        os.replace(tmp,target_dir)
        return
    bak=target_dir+".bak"
    old_dir=None
    if not os.path.exists(bak):
        # First replacement: retain the current generation as the recovery
        # directory until the new generation has been published.
        os.replace(target_dir,bak)
    else:
        # A backup already exists and may be the only valid generation.
        # Never delete it before the new checkpoint is safely in place.
        old_dir=target_dir+".old."+uuid.uuid4().hex
        os.replace(target_dir,old_dir)
    try:
        os.replace(tmp,target_dir)
    except BaseException:
        if old_dir is not None:
            if not os.path.exists(target_dir):
                os.replace(old_dir,target_dir)
        elif os.path.exists(bak) and not os.path.exists(target_dir):
            os.replace(bak,target_dir)
        raise
    if old_dir is not None:
        shutil.rmtree(old_dir,ignore_errors=True)
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
    _check_optimizer_state(params,opt_state)
    return _write_tree(target_dir,manifest,params,opt_state)


def _check_consistency(manifest,params):
    from . import config as _cfg

    preset = manifest.get("preset", "")
    if preset not in _cfg.PRESETS:
        raise ValueError(f"unknown checkpoint preset {preset!r}")
    spec = _cfg.PRESETS[preset]
    n = manifest.get("n_experts_current")
    if spec.kind == "moe":
        if n is None:
            n = spec.n_experts_init
        if isinstance(n, bool) or not isinstance(n, int) or n < spec.top_k:
            raise ValueError("invalid n_experts_current for MoE preset")
        expected_eids = list(range(n))
        registry = manifest.get("expert_registry", [])
        if registry and registry != expected_eids:
            raise ValueError("expert registry disagrees with dense expert indices")
        arch = manifest.get("arch", {})
        if arch.get("d_model", spec.d_model) != spec.d_model:
            raise ValueError("manifest d_model disagrees with selected preset")
        if arch.get("n_layers", spec.n_layers) != spec.n_layers:
            raise ValueError("manifest n_layers disagrees with selected preset")
        if arch.get("expert_hidden", spec.expert_hidden) != spec.expert_hidden:
            raise ValueError("manifest expert_hidden disagrees with selected preset")
    else:
        if n not in (None, 0):
            raise ValueError("dense checkpoint cannot contain MoE experts")
        n = 0
        arch = manifest.get("arch", {})
        if arch.get("d_model", spec.d_model) != spec.d_model:
            raise ValueError("manifest d_model disagrees with selected preset")
        if arch.get("n_layers", spec.n_layers) != spec.n_layers:
            raise ValueError("manifest n_layers disagrees with selected preset")

    expected = {
        "embed": (spec.vocab_size, spec.d_model),
        "final_norm": (spec.d_model,),
    }
    for layer in range(spec.n_layers):
        prefix = f"L{layer}."
        for name in ("Wr", "Wk", "Wv", "Wo"):
            expected[prefix + "attn." + name] = (spec.d_model, spec.d_model)
        for name in ("w", "u"):
            expected[prefix + "attn." + name] = (spec.d_model,)
        expected[prefix + "n1"] = (spec.d_model,)
        expected[prefix + "n2"] = (spec.d_model,)
        if spec.kind == "dense":
            expected[prefix + "ffn.Wa"] = (spec.d_model, spec.ffn_hidden)
            expected[prefix + "ffn.Wb"] = (spec.d_model, spec.ffn_hidden)
            expected[prefix + "ffn.Wc"] = (spec.ffn_hidden, spec.d_model)
    if spec.kind == "moe":
        expected["router"] = (spec.d_model, n)
        for eid in range(n):
            expected[f"expert.{eid}.Win"] = (spec.d_model, spec.expert_hidden)
            expected[f"expert.{eid}.Wout"] = (spec.expert_hidden, spec.d_model)

    actual_keys = set(params)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"checkpoint parameter registry mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    for key, shape in expected.items():
        value = params[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint parameter {key} is not a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"checkpoint parameter {key} has shape {tuple(value.shape)}, expected {shape}"
            )
    vocab = manifest.get("vocab", {})
    if vocab and vocab.get("vocab_size", spec.vocab_size) != spec.vocab_size:
        raise ValueError("checkpoint vocabulary size disagrees with selected preset")
    arch = manifest.get("arch", {})
    if arch.get("vocab_size", spec.vocab_size) != spec.vocab_size:
        raise ValueError("manifest vocab_size disagrees with selected preset")



def _check_optimizer_state(params,opt_state):
    """Reject optimizer buffers that cannot safely update the loaded tensors."""
    if not isinstance(opt_state, dict):
        raise ValueError("optimizer state must be a mapping")
    for name, state in opt_state.items():
        if name not in params:
            raise ValueError(f"optimizer state references missing parameter {name!r}")
        if not isinstance(state, dict):
            raise ValueError(f"optimizer state for {name} must be a mapping")
        shape = tuple(params[name].shape)
        for slot in ("m", "v"):
            value = state.get(slot)
            if value is not None:
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                    got = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                    raise ValueError(
                        f"optimizer {slot} shape for {name} is {got}, expected {shape}"
                    )
        factored = "R" in state or "C" in state
        if factored:
            if len(shape) != 2:
                raise ValueError(f"factored optimizer state attached to non-matrix parameter {name}")
            for slot, expected in (("R", (shape[0],)), ("C", (shape[1],))):
                value = state.get(slot)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                    got = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                    raise ValueError(
                        f"optimizer {slot} shape for {name} is {got}, expected {expected}"
                    )



def _load_checkpoint_dir(ckpt_dir):
    mp=os.path.join(ckpt_dir,"manifest.json")
    if not os.path.exists(mp):raise FileNotFoundError(f"no manifest at {mp}")
    with open(mp) as f:man=json.load(f)
    version=man.get("version")
    if version not in (CHECKPOINT_VERSION,LEGACY_VERSION):raise ValueError(f"incompatible checkpoint version {version!r}")
    if version==CHECKPOINT_VERSION:
        names=("params.pt","optim.pt")
        for name in names:
            path=os.path.join(ckpt_dir,name)
            if not os.path.exists(path):raise FileNotFoundError(f"checkpoint missing {name} (incomplete write?)")
            expected_hash=man.get("files",{}).get(name)
            if not isinstance(expected_hash,str) or expected_hash!=_sha256(path):raise ValueError(f"missing or invalid integrity hash for {name}")
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
    _check_optimizer_state(params,opt_state)
    return {"manifest":man,"params":params,"opt_state":opt_state,"dir":ckpt_dir}



def load_checkpoint(ckpt_dir):
    """Load the primary checkpoint, falling back to a valid backup on corruption."""
    primary=os.path.normpath(ckpt_dir)
    backup=primary+".bak"
    candidates=[]
    for candidate in (primary,backup):
        if os.path.isfile(os.path.join(candidate,"manifest.json")):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint manifest at {os.path.join(primary, 'manifest.json')} "
            f"or {os.path.join(backup, 'manifest.json')}"
        )
    failures=[]
    for candidate in candidates:
        try:
            return _load_checkpoint_dir(candidate)
        except Exception as exc:
            failures.append((candidate,exc))
    details="; ".join(f"{path}: {error}" for path,error in failures)
    raise RuntimeError(f"no valid checkpoint found (primary/backup recovery failed): {details}") from failures[-1][1]
