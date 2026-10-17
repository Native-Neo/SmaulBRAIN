import torch

from smaulbrain.growth import eval_loss_on_tokens, eval_loss_on_tokens_disabled


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def forward(self, tokens, train=False, disabled=()):
        values = list(tokens)
        self.calls.append((values, tuple(disabled)))
        logits = torch.zeros((len(values), 259), dtype=torch.float32)
        return logits, None


def test_evaluation_forwards_only_tokens_that_have_next_token_targets():
    model = _RecordingModel()
    sequence = [11, 12, 13, 14, 15, 16]

    loss = eval_loss_on_tokens(model, sequence)
    assert model.calls == [([11, 12, 13, 14, 15], ())]
    assert torch.isfinite(torch.tensor(loss))

    model.calls.clear()
    masked_loss = eval_loss_on_tokens_disabled(model, sequence, disabled=7)
    assert model.calls == [([11, 12, 13, 14, 15], (7,))]
    assert torch.isfinite(torch.tensor(masked_loss))


def test_evaluation_requires_at_least_one_prediction_target():
    import pytest

    with pytest.raises(ValueError, match="at least two tokens"):
        eval_loss_on_tokens(_RecordingModel(), [5])
