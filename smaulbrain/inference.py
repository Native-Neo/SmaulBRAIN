"""Inference engine using Torch tensors and persistent recurrent state."""
import torch


class InferenceEngine:
    def __init__(self, model, seed=0, temperature=1.0, top_k=0, top_p=1.0):
        self.model = model
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        if not 0 < self.top_p <= 1:
            raise ValueError("--top-p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("--top-k must be >= 0")
        if self.temperature < 0:
            raise ValueError("--temperature must be >= 0")
        self.generator = torch.Generator(device=getattr(model, "device", "cpu"))
        self.generator.manual_seed(int(seed))
        self.states = model.init_states()
        self._next_logits = None

    def reset(self):
        self.states = self.model.init_states()
        self._next_logits = None

    def export_states(self):
        return self.model.serialize_states(self.states)

    def import_states(self, blob):
        self.states = [
            torch.as_tensor(blob[str(i)], dtype=torch.float32, device=self.model.device)
            for i in range(len(blob))
        ]
        self._next_logits = None

    def prefill(self, tokens, chunk=0):
        """Consume context once and retain logits predicting its next token."""
        toks = list(tokens)
        if not toks:
            return self.states

        size = chunk if chunk and chunk > 0 else len(toks)
        for i in range(0, len(toks), size):
            logits, _ = self.model.forward(toks[i:i + size], states=self.states)
            if logits.shape[0]:
                self._next_logits = logits[-1].detach()
        return self.states

    def _sample(self, logits):
        if self.temperature == 0:
            return int(torch.argmax(logits).item())
        l = logits.float() / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, l.numel())
            threshold = torch.topk(l, k).values[-1]
            l = l.masked_fill(l < threshold, -torch.inf)
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
        """Yield tokens; prompt prefill logits are used directly, without replaying the prompt's last token."""
        self.prefill(prompt_tokens)
        stop = set(stop_ids)
        for _ in range(max_new_tokens):
            if self._next_logits is None:
                # Empty context has no next-token logits. Seed the recurrent
                # state with byte 0 once, matching the previous empty-prompt behavior.
                self._next_logits = self.model.step_token(0, self.states)
            nxt = self._sample(self._next_logits)
            yield nxt
            if nxt in stop:
                break
            self._next_logits = self.model.step_token(nxt, self.states)
