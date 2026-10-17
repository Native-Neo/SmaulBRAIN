from smaulbrain import config as C


def test_moe_parameter_totals_include_router_growth():
    spec = C.PRESETS["moe-tiny"]
    initial = C.arch_report(spec, spec.n_experts_init)
    grown = C.arch_report(spec, spec.n_experts_init + 2)

    assert initial["router_params"] == spec.d_model * spec.n_experts_init
    assert grown["router_params"] == spec.d_model * (spec.n_experts_init + 2)
    assert grown["total_params"] - initial["total_params"] == (
        2 * C.expert_param_count(spec) + 2 * spec.d_model
    )
    assert grown["total_params"] == C.whole_model_total(
        spec, spec.n_experts_init + 2
    )
