import torch

from smaulbrain import config as C
from smaulbrain import model as M
from smaulbrain import paging as P


def test_paged_gradient_transfer_reuses_gradient_storage(tmp_path):
    spec = C.PRESETS["moe-tiny"]
    params = M.build_params(spec, torch.Generator().manual_seed(3))
    for eid in range(spec.n_experts_init):
        torch.save(
            {
                "Win": params[f"expert.{eid}.Win"].detach().cpu(),
                "Wout": params[f"expert.{eid}.Wout"].detach().cpu(),
            },
            tmp_path / f"expert_{eid}.pt",
        )

    store = P.ExpertStore(
        "D2R",
        spec.n_experts_init,
        disk_dir=str(tmp_path),
        ram_cache_max_experts=8,
        staging_max_bytes=1 << 26,
    )
    model = M.Model(
        spec,
        params,
        paging_mode="D2R",
        paging_store=store,
        capacity_factor=99.0,
        device="cpu",
    )

    loss, grads, _ = model.loss_and_backward(list(range(17)))

    assert torch.isfinite(torch.tensor(loss))
    compared = 0
    for (eid, suffix), leaf in model._paged_leaves.items():
        if leaf.grad is None:
            continue
        name = f"expert.{eid}.{suffix}"
        assert name in grads
        assert (
            grads[name].untyped_storage().data_ptr()
            == leaf.grad.untyped_storage().data_ptr()
        )
        compared += 1
    assert compared > 0
