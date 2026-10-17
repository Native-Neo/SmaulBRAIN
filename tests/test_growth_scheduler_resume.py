from smaulbrain.growth import GrowthScheduler


def test_growth_evidence_threshold_survives_checkpoint_round_trip():
    scheduler = GrowthScheduler(
        every_steps=0,
        loss_nats=None,
        loss_beta=0.5,
        min_eval_samples=17,
    )
    restored = GrowthScheduler.from_record(scheduler.to_record())

    assert restored.min_eval_samples == 17
    assert restored.every_steps == 0
    assert restored.loss_threshold is None
    assert restored.loss_beta == 0.5
