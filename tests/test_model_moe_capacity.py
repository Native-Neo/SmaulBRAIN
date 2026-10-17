import types

import torch

from smaulbrain import config as C
from smaulbrain import model as M


def test_moe_does_not_evaluate_capacity_dropped_assignments():
    spec = C.PRESETS["moe-tiny"]
    rng = torch.Generator(device="cpu").manual_seed(0)
    params = M.build_params(spec, rng, spec.n_experts_init, device="cpu")
    model = M.Model(spec, params, capacity_factor=0.1)
    with torch.no_grad():
        model.p["router"].zero_()
        model.p["router"][:, 0].fill_(2.0)
        model.p["router"][:, 1].fill_(1.0)

    calls = []

    def fake_expert(self, values, eid):
        calls.append((eid, values.shape[0]))
        return torch.ones_like(values)

    model._expert = types.MethodType(fake_expert, model)
    output, info = model._moe(torch.ones(3, spec.d_model))

    assert calls == [(0, 1), (1, 1)]
    assert info["overflow_dropped"] == 4
    assert torch.allclose(output[0], torch.ones(spec.d_model))
    assert torch.count_nonzero(output[1:]).item() == 0
