import torch

from smaulbrain.inference import InferenceEngine


class _CountingModel:
    device = torch.device("cpu")

    def __init__(self):
        self.step_inputs = []

    def init_states(self):
        return [0]

    def forward(self, tokens, states=None):
        start = states[0]
        logits = torch.full((len(tokens), 8), -100.0)
        for i in range(len(tokens)):
            logits[i, start + i + 1] = 100.0
        states[0] += len(tokens)
        return logits, states

    def step_token(self, token, states):
        self.step_inputs.append(token)
        states[0] += 1
        logits = torch.full((8,), -100.0)
        logits[states[0]] = 100.0
        return logits


def test_generation_uses_last_prefill_logits_without_replaying_prompt():
    model = _CountingModel()
    engine = InferenceEngine(model, temperature=0.0)

    assert engine.generate([40, 41], max_new_tokens=3) == [2, 3, 4]
    assert model.step_inputs == [2, 3]


def test_chunked_prefill_preserves_next_token_logits():
    model = _CountingModel()
    engine = InferenceEngine(model, temperature=0.0)

    engine.prefill([40, 41, 42], chunk=2)
    assert list(engine.generate_stream([], max_new_tokens=2)) == [3, 4]
    assert model.step_inputs == [3]
