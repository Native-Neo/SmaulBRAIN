"""PyTorch tensor operators used by SmaulBRAIN."""
import torch

EPS = 1e-6
MAX_TMP_ELEMENTS = 1 << 26


def _tensor(x, device=None):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32, device=device)


def rmsnorm_fwd(x, g, eps=EPS):
    x = _tensor(x).float(); g = _tensor(g, device=x.device).float()
    ms = x.square().mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(ms + eps)
    return xn * g, (x, g, xn, ms)


def rmsnorm_bwd(dy, cache, eps=EPS):
    x, g, xn, ms = cache
    dy = _tensor(dy, device=x.device).float(); n = x.shape[-1]
    inv = torch.rsqrt(ms + eps)
    dxn = dy * g
    dot = (dxn * x).sum(dim=-1, keepdim=True)
    dx = (dxn - x * dot / (n * (ms + eps))) * inv
    dims = tuple(range(dy.ndim - 1))
    dg = (dy * xn).sum(dim=dims) if dims else dy * xn
    return dx, dg


def sigmoid(x):
    return torch.sigmoid(x)


def silu(x):
    return torch.nn.functional.silu(x)


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def gelu_tanh_grad(x):
    c = 0.7978845608028654
    z = c * (x + 0.044715 * x.pow(3))
    t = torch.tanh(z)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t.square()) * c * (1 + 0.134145 * x.square())


def softplus(x):
    return torch.nn.functional.softplus(x)


def scatter_add_rows(dst, ids, src):
    ids = torch.as_tensor(ids, dtype=torch.long, device=dst.device).reshape(-1)
    dst.index_add_(0, ids, src)
    return dst


class RecurrentAttention:
    """Standalone PyTorch recurrent-attention block used by math tests/tools."""
    def __init__(self, d, rng=None, device="cpu"):
        import math
        if rng is None:
            rand = lambda shape: torch.randn(shape, device=device) / math.sqrt(d)
            self.w = torch.randn(d, device=device) * 0.1
        else:
            rand = lambda shape: torch.as_tensor(rng.standard_normal(shape) / math.sqrt(d), dtype=torch.float32, device=device)
            self.w = torch.as_tensor(rng.standard_normal(d) * 0.1, dtype=torch.float32, device=device)
        self.Wr = rand((d, d)); self.Wk = rand((d, d)); self.Wv = rand((d, d)); self.Wo = rand((d, d))
        self.u = torch.zeros(d, device=device)

    def forward(self, X, s0=None):
        X = _tensor(X, device=self.Wr.device).float()
        Wr, Wk, Wv, Wo = (_tensor(getattr(self, k), device=X.device).float() for k in ("Wr", "Wk", "Wv", "Wo"))
        w = _tensor(self.w, device=X.device).float(); u = _tensor(self.u, device=X.device).float()
        alpha = torch.exp(-torch.nn.functional.softplus(w))
        state = torch.zeros_like(alpha) if s0 is None else _tensor(s0, device=X.device).float().clone()
        outputs = []
        for row in X:
            r, k, v = row @ Wr, row @ Wk, row @ Wv
            kv = k * v; state = alpha * state + kv
            outputs.append(torch.sigmoid(r) * (state + u * kv) @ Wo)
        Y = torch.stack(outputs) if outputs else X.new_zeros((0, Wo.shape[1]))
        return Y, (state.clone(), (X,))
