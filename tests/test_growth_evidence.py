import torch

from smaulbrain.growth import rank_experts


class _ApproximateRankModel:
    def __init__(self):
        self.p = {"expert.0.Win": torch.zeros(1)}
        self.calls = 0

    def forward(self, tokens, train=False, disabled=()):
        self.calls += 1
        logits = torch.zeros((len(tokens), 16))
        if disabled:
            logits[:, int(disabled[0])] = 1.0
        return logits, None


def test_ranking_uses_all_batches_for_evidence_but_scores_subset():
    batches = [list(range(start, start + 5)) for start in (0, 5, 10, 0)]
    model = _ApproximateRankModel()

    scores, samples = rank_experts(
        model,
        batches,
        min_samples=10,
        max_batches=1,
    )

    assert samples == 16
    assert scores is not None and set(scores) == {0}
    # One baseline and one masked score; the remaining batches gate evidence
    # but do not incur extra forward passes.
    assert model.calls == 2
