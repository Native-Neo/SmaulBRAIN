import torch

from smaulbrain.growth import rank_experts


class _ModelThatMustNotBeRanked:
    p = {"expert.0.Win": torch.zeros(1)}

    def forward(self, *args, **kwargs):
        raise AssertionError("ranking should stop before scoring insufficient evidence")


def test_rank_experts_does_not_count_ignored_batches_as_evidence():
    batches = [[1, 2, 3, 4, 5] for _ in range(4)]

    scores, samples = rank_experts(
        _ModelThatMustNotBeRanked(),
        batches,
        min_samples=10,
        max_batches=1,
    )

    assert scores is None
    assert samples == 4
