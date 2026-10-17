"""Optional CPU C++ fused SmaulOPT update reached through Torch tensor views."""
import torch
from . import _native_opt_numpy as _backend


def factored_step(opt,w,g,st,collect_norms):
    # The native ABI accepts float* only. Fall back to Torch for any other
    # dtype instead of calling Tensor.numpy() on bfloat16 or misreading memory
    # as float32 in the C++ kernel.
    if w.device.type != "cpu" or w.dtype != torch.float32 or not w.is_contiguous() or w.ndim != 2 or w.numel() < 1024:
        return None
    if g.device.type != "cpu" or g.dtype != torch.float32 or g.shape != w.shape or not g.is_contiguous():
        return None
    if any(not isinstance(st[k],torch.Tensor) or st[k].device.type != "cpu"
           or st[k].dtype != torch.float32 or not st[k].is_contiguous()
           for k in ("m","R","C")):
        return None
    if st["m"].shape != w.shape or st["R"].shape != (w.shape[0],) or st["C"].shape != (w.shape[1],): return None
    return _backend.factored_step(opt,w.detach().numpy(),g.detach().contiguous().numpy(),
                                  {k:st[k].detach().numpy() for k in ("m","R","C")},collect_norms)
