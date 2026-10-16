"""SmaulOPT: factored-V adaptive optimizer.

Per parameter tensor with gradient g (fp32):

  m <- beta1 * m + (1 - beta1) * g                      (first moment)
  if W is 2-D (m x n):
      R <- beta2 * R + (1 - beta2) * mean(g^2, axis=1)  (row statistic, size m)
      C <- beta2 * C + (1 - beta2) * mean(g^2, axis=0)  (col statistic, size n)
      V_hat = outer(R, C) / max(mean(R), eps_tiny)       (rank-1 reconstruction)
  else (vectors / scalars / unusual shapes):
      v <- beta2 * v + (1 - beta2) * g^2                 (full statistic;
            documented as not-factorized for non-matrix shapes)
  No bias correction is applied (documented choice); epsilon is added
  outside the sqrt: denom = sqrt(V_hat) + eps.
  Decoupled weight decay: W <- W - lr * wd * W (before adaptive step).
  Update: W <- W - lr * m / denom, computed in float32/float64 accumulation
  with fp32 state storage.

Missing gradients (None) and unused experts are skipped safely.
"""
import numpy as np

from . import _native_opt


class SmaulOPT:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.98, eps=1e-8, weight_decay=0.0):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.state = {}  # name -> {"m":..., "R":..., "C":...} or {"m":..., "v":...}
        self.n_updates = 0

    def _init_state(self, name: str, w: np.ndarray):
        if w.ndim == 2:
            self.state[name] = {
                "m": np.zeros_like(w),
                "R": np.zeros(w.shape[0], dtype=np.float32),
                "C": np.zeros(w.shape[1], dtype=np.float32),
                "factored": True,
            }
        else:
            self.state[name] = {
                "m": np.zeros_like(w),
                "v": np.zeros_like(w),
                "factored": False,
            }

    def register(self, params: dict) -> None:
        for k, w in params.items():
            if k not in self.state:
                self._init_state(k, w)

    def drop(self, names) -> None:
        for k in names:
            self.state.pop(k, None)

    def state_memory_elements(self) -> dict:
        factored, full = 0, 0
        for s in self.state.values():
            if s["factored"]:
                factored += s["m"].size + s["R"].size + s["C"].size
            else:
                full += 2 * s["m"].size
        return {"factored_elements": factored, "full_elements": full}

    def step(self, params: dict, grads: dict, collect_norms: bool = True) -> dict:
        """In-place update; optionally return per-parameter RMS update norms."""
        norms = {}
        beta1 = np.float32(self.beta1)
        beta2 = np.float32(self.beta2)
        one_minus_beta1 = np.float32(1.0 - self.beta1)
        one_minus_beta2 = np.float32(1.0 - self.beta2)
        eps = np.float32(self.eps)
        for name, w in params.items():
            g = grads.get(name)
            if g is None:
                continue
            if name not in self.state:
                self._init_state(name, w)
            st = self.state[name]
            if st["factored"]:
                native = _native_opt.factored_step(self, w, g, st, collect_norms)
                if native is not None:  # fused native update (see _optimizer_kernels.cpp)
                    if collect_norms:
                        norms[name] = native
                    continue
            gf = np.asarray(g, dtype=np.float32)
            if self.weight_decay:
                w -= np.float32(self.lr * self.weight_decay) * w

            m = st["m"]
            if m.dtype == np.float32 and m.flags.writeable:
                m *= beta1                      # in place: same values, no new matrix
                m += gf * one_minus_beta1
            else:
                m = (m * beta1 + gf * one_minus_beta1).astype(np.float32, copy=False)
                st["m"] = m
            g2 = gf * gf
            if st["factored"]:
                row = np.mean(g2, axis=1, dtype=np.float32)
                col = np.mean(g2, axis=0, dtype=np.float32)
                del g2  # release the full matrix-square temporary before updates
                R = st["R"] * beta2 + row * one_minus_beta2
                C = st["C"] * beta2 + col * one_minus_beta2
                st["R"] = R.astype(np.float32, copy=False)
                st["C"] = C.astype(np.float32, copy=False)
                mean_r = float(np.mean(R, dtype=np.float64)) if R.size else 0.0
                denom_r = np.float32(mean_r if mean_r > 1e-30 else 1e-30)
                # Reuse the outer-product buffer as sqrt(vhat), then as the
                # update itself; avoid an extra full-matrix m / denom allocation.
                upd = np.outer(R, C)
                upd /= denom_r
                np.maximum(upd, 0.0, out=upd)
                np.sqrt(upd, out=upd)
                upd += eps
                np.divide(m, upd, out=upd)
            else:
                v = st["v"] * beta2 + g2 * one_minus_beta2
                del g2
                st["v"] = v.astype(np.float32, copy=False)
                upd = np.maximum(v, 0.0)
                np.sqrt(upd, out=upd)
                upd += eps
                np.divide(m, upd, out=upd)
            if collect_norms:
                # Only a scalar reduction uses fp64; never widen the update tensor.
                norms[name] = float(np.sqrt(np.mean(upd * upd, dtype=np.float64)))
            np.multiply(upd, np.float32(self.lr), out=upd)
            np.subtract(w, upd, out=w)
        self.n_updates += 1
        return norms

    # -- persistence --
    def state_dict(self) -> dict:
        return {k: {sk: sv.copy() for sk, sv in s.items() if not isinstance(sv, bool)}
                for k, s in self.state.items()}

    def load_state_dict(self, sd: dict, factored_flags=None) -> None:
        self.state = {}
        for k, s in sd.items():
            entry = {sk: sv.copy() for sk, sv in s.items()}
            entry["factored"] = bool(
                factored_flags.get(k, ("R" in entry)) if factored_flags else ("R" in entry))
            self.state[k] = entry
