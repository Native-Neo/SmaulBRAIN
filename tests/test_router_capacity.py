import torch

from smaulbrain.moe import Router


def test_capacity_keeps_strongest_assignments_not_earliest_tokens():
    router = Router(1, 2)
    router.W = torch.tensor([[2.0, 1.0]])

    x = torch.tensor([[1.0], [10.0], [2.0]])
    indices, weights, info, cache = router.route(x, top_k=1, capacity_factor=0.34)
    keep = cache[-1].reshape(-1)

    assert indices[:, 0].tolist() == [0, 0, 0]
    assert keep.tolist() == [False, True, False]
    assert weights[:, 0].tolist() == [0.0, 1.0, 0.0]
    assert info["overflow_dropped"] == 2
    assert info["cap_per_expert"] == 1
