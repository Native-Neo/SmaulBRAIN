"""Inference engine using Torch tensors and persistent recurrent state."""
import math

import torch


class InferenceEngine:
    def __init__(self, model, seed=0, temperature=1.0, top_k=0, top_p=1.0):
        self.model = model
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("--top-p must be finite and in (0, 1]")
        if self.top_k < 0:
            raise ValueError("--top-k must be >= 0")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("--temperature must be finite and >= 0")
        self.generator = torch.Generator(device=getattr(model, "device", "cpu"))
        self.generator.manual_seed(int(seed))
        self.states = model.init_states()
        self._next_logits = None
        self._pending_token = None

    def reset(self):
        self.states = self.model.init_states()
        self._next_logits = None
        self._pending_token = None

    def export_states(self):
        return self.model.serialize_states(self.states)

    def import_states(self, blob):
        self.states = [
            torch.as_tensor(blob[str(i)], dtype=torch.float32, device=self.model.device)
            for i in range(len(blob))
        ]
        self._next_logits = None
        self._pending_token = None

    def prefill(self, tokens, chunk=0):
        """Consume context once and retain logits predicting its next token."""
        # A completed/partially consumed generation leaves its final sampled
        # token pending so we can avoid one wasted model call at the end. Consume
        # it only when the caller actually continues or appends more context.
        if self._pending_token is not None:
            self._next_logits = self.model.step_token(self._pending_token, self.states)
            self._pending_token = None
        toks = list(tokens)
        if not toks:
            return self.states

        self._next_logits = None
        size = chunk if chunk and chunk > 0 else len(toks)
        for i in range(0, len(toks), size):
            logits, _ = self.model.forward(toks[i:i + size], states=self.states)
            if logits is not None and getattr(logits, "ndim", 0) >= 1 and logits.shape[0]:
                self._next_logits = logits[-1].detach()

        if self._next_logits is None:
            # Compatibility path for models/stubs whose forward() advances
            # recurrent state but does not return logits: prefill all but the
            # final token, then consume that token exactly once to predict.
            self.states = self.model.init_states()
            prefix = toks[:-1]
            for i in range(0, len(prefix), size):
                self.model.forward(prefix[i:i + size], states=self.states)
            self._next_logits = self.model.step_token(toks[-1], self.states)
        return self.states

    def _sample(self, logits):
        logits = torch.as_tensor(logits)
        if logits.ndim != 1 or logits.numel() == 0:
            raise ValueError("sampling logits must be a non-empty 1-D tensor")
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("cannot sample from non-finite model logits")
        if self.temperature == 0:
            return int(torch.argmax(logits).item())
        # Subtract the maximum before temperature scaling so even very small
        # positive temperatures cannot overflow the maximum logit to +inf.
        l = logits.double()
        l = (l - l.max()) / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, l.numel())
            # Mask by rank, not by cutoff value: cutoff-based filtering
            # retains every tied logit and can silently exceed top_k.
            order = torch.argsort(l, descending=True, stable=True)
            l = l.clone()
            l[order[k:]] = -torch.inf
        if self.top_p < 1:
            sorted_l, idx = torch.sort(l, descending=True)
            probs = torch.softmax(sorted_l, dim=0)
            cumulative = probs.cumsum(0)
            remove = cumulative - probs >= self.top_p
            sorted_l = sorted_l.masked_fill(remove, -torch.inf)
            l = torch.full_like(l, -torch.inf).scatter(0, idx, sorted_l)
        probs = torch.softmax(l, dim=0)
        return int(torch.multinomial(probs, 1, generator=self.generator).item())

    def generate(self, prompt_tokens, max_new_tokens=64, stop_ids=()):
        return list(self.generate_stream(prompt_tokens, max_new_tokens, stop_ids))

    def generate_stream(self, prompt_tokens, max_new_tokens=64, stop_ids=()):
        """Yield tokens without replaying the prompt or advancing past the final output."""
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be an integer >= 0")
        if max_new_tokens == 0:
            return
        self.prefill(prompt_tokens)
        stop = set(stop_ids)
        for index in range(max_new_tokens):
            if self._next_logits is None:
                # Empty context has no next-token logits. Seed the recurrent
                # state with byte 0 once, matching the previous empty-prompt behavior.
                self._next_logits = self.model.step_token(0, self.states)
            nxt = self._sample(self._next_logits)
            # Record before yielding so early consumer cancellation still
            # leaves recurrent state resumable. The token is cleared only if
            # the generator continues and actually advances the model through it.
            self._pending_token = nxt
            yield nxt
            if nxt in stop or index + 1 == max_new_tokens:
                break
            self._next_logits = self.model.step_token(nxt, self.states)
            self._pending_token = None
