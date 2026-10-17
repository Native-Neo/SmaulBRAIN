"""Optional CPU C++ fused SmaulOPT update reached through Torch tensor views."""
import torch
from . import _native_opt_numpy as _backend


def factored_step(opt,w,g,st,collect_norms):
    if w.device.type != "cpu" or not w.is_contiguous() or w.ndim != 2 or w.numel() < 1024:
        return None
    if any(not isinstance(st[k],torch.Tensor) or st[k].device.type != "cpu" or not st[k].is_contiguous() for k in ("m","R","C")):
        return None
    if st["m"].shape != w.shape or st["R"].shape != (w.shape[0],) or st["C"].shape != (w.shape[1],): return None
    return _backend.factored_step(opt,w.detach().numpy(),g.detach().contiguous().numpy(),
                                  {k:st[k].detach().numpy() for k in ("m","R","C")},collect_norms)
