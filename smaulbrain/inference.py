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
            k = min(self.top_k, l.size)  # argpartition rejects k > vocab
            idx = np.argpartition(l, -k)[-k:]
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

    def generate(self, prompt_tokens, max_new_tokens=64, stop_ids=()):
        # Keep the final input token in a scalar. Rebuilding prompt_tokens + out
        # on every step made decoding allocate and copy O(prompt + generated)^2
        # elements over a long generation.
        prompt = list(prompt_tokens)
        self.prefill(prompt)
        out = []
        last = prompt[-1] if prompt else 0
        stop_set = set(stop_ids)
        for _ in range(max_new_tokens):
            logits = self.model.step_token(last, self.states)
            nxt = self._sample(logits)
            out.append(nxt)
            last = nxt
            if nxt in stop_set:
                break
        return out

    def generate_stream(self, prompt_tokens, max_new_tokens=64, stop_ids=()):
        prompt = list(prompt_tokens)
        self.prefill(prompt)
        out = []
        last = prompt[-1] if prompt else 0
        stop_set = set(stop_ids)
        for _ in range(max_new_tokens):
            logits = self.model.step_token(last, self.states)
            nxt = self._sample(logits)
            out.append(nxt)
            last = nxt
            yield nxt
            if nxt in stop_set:
                break
