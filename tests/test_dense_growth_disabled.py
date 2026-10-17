import types

from smaulbrain import config as C
from smaulbrain import data as D
from smaulbrain import train_loop as T


def test_dense_trainer_disables_moe_growth_triggers(tmp_path):
    args = types.SimpleNamespace(
        paging="R",
        paging_cache_experts=8,
        paging_staging_bytes=1 << 26,
        forward_quant="fp32",
        backward_mode="fp32",
        device="cpu",
        checkpoint=str(tmp_path / "dense-run"),
        seed=0,
        seq_len=16,
        lr=3e-4,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        weight_decay=0.0,
        growth_every_steps=1,
        growth_loss_nats=100.0,
        growth_loss_beta=0.98,
        growth_min_eval_samples=16,
        growth_noise_scale=0.01,
        growth_seed=1,
        expert_capacity_factor=1.25,
    )
    trainer = T.Trainer(
        C.PRESETS["dense-tiny"], args, D.ByteStream(data=D.synth_bytes(128, 0))
    )

    assert trainer.sched.every_steps == 0
    assert trainer.sched.next_periodic_step is None
    assert trainer.sched.loss_threshold is None
    history = trainer.train(max_steps=1, log_every=1)
    assert not any(e.get("event") == "growth-skipped-dense" for e in history)
