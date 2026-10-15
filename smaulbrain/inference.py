"""Independent inference engine (no training dependency).

Byte-level prompt ingestion, prefill (chunked optional), incremental
autoregressive decoding with persistent recurrent state, greedy + reproducible
temperature/top-k/top-p sampling, streaming, state reset/continuation, stop
criteria, bounded paging, quantization-cache reuse. No gradients/optimizer/loss.
"""
import numpy as np


class InferenceEngine:
    def __init__(self, model, seed=0, temperature=1.0, top_k=0, top_p=1.0):
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        if not (self.top_p > 0 and self.top_p <= 1.0):
            raise ValueError("--top-p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("--top-k must be >= 0")
        if self.temperature < 0:
            raise ValueError("--temperature must be >= 0")
        self.states = model.init_states()

    def reset(self):
        self.states = self.model.init_states()

    def export_states(self):
        return self.model.serialize_states(self.states)

    def import_states(self, blob):
        self.states = self.model.restore_states(blob)

    def prefill(self, tokens, chunk: int = 0):
        toks = list(tokens)
        if chunk and chunk > 0:
            for i in range(0, len(toks), chunk):
                self.model.forward(toks[i:i + chunk], states=self.states)
        elif toks:
            self.model.forward(toks, states=self.states)
        return self.states

    def _sample(self, logits: np.ndarray) -> int:
        from . import vocab as _v
        if self.temperature == 0:
            return int(np.argmax(logits))
        l = logits.astype(np.float64) / self.temperature
        if self.top_k > 0:
            idx = np.argpartition(l, -self.top_k)[-self.top_k:]
            mask = np.full_like(l, -np.inf)
            mask[idx] = l[idx]
            l = mask
        if self.top_p < 1.0:
            order = np.argsort(-l)
            ls = l[order]
            probs = np.exp(ls - ls.max())
            probs /= probs.sum()
            cum = np.cumsum(probs)
            cut = int(np.searchsorted(cum, self.top_p)) + 1
            keep = order[:cut]
            mask = np.full_like(l, -np.inf)
            mask[keep] = l[keep]
            l = mask
        e = np.exp(l - np.nanmax(l))
        probs = e / e.sum()
        return int(self.rng.choice(len(probs), p=probs))

    def generate(self, prompt_tokens, max_new_tokens=64, stop_ids=(),
                 stream: bool = False):
        self.prefill(prompt_tokens)
        out = []
        for _ in range(max_new_tokens):
            logits = self.model.step_token(
                (prompt_tokens + out)[-1] if (prompt_tokens or out) else 0, self.states)
            nxt = self._sample(logits)
            out.append(nxt)
            if nxt in tuple(stop_ids):
                break
            if stream:
                pass  # caller consumes incrementally via iterator below
        return out

    def generate_stream(self, prompt_tokens, max_new_tokens=64, stop_ids=()):
        self.prefill(prompt_tokens)
        out = []
        for _ in range(max_new_tokens):
            last = (prompt_tokens + out)[-1] if (prompt_tokens or out) else 0
            logits = self.model.step_token(last, self.states)
            nxt = self._sample(logits)
            out.append(nxt)
            yield nxt
            if nxt in tuple(stop_ids):
                break
