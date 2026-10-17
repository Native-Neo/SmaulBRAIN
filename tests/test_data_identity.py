import types

import pytest

from smaulbrain import config as C
from smaulbrain import data as D
from smaulbrain import train_loop as T


def _trainer(checkpoint, stream):
    args = types.SimpleNamespace(
        paging="R",
        paging_cache_experts=8,
        paging_staging_bytes=1 << 26,
        forward_quant="fp32",
        backward_mode="fp32",
        device="cpu",
        checkpoint=str(checkpoint),
        seed=0,
        seq_len=16,
        lr=3e-4,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        weight_decay=0.0,
        growth_every_steps=0,
        growth_loss_nats=None,
        growth_loss_beta=0.98,
        growth_min_eval_samples=16,
        growth_noise_scale=0.01,
        growth_seed=1,
        expert_capacity_factor=99.0,
    )
    return T.Trainer(C.PRESETS["moe-tiny"], args, stream)


def test_resume_rejects_different_in_memory_corpus_with_same_size(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    first = D.ByteStream(data=D.synth_bytes(4096, 0))
    trainer = _trainer(checkpoint, first)
    trainer.train(max_steps=1)
    trainer.save()

    different = D.ByteStream(data=D.synth_bytes(4096, 1))
    resumed = _trainer(checkpoint, different)
    with pytest.raises(ValueError, match="contents or file metadata changed"):
        resumed.load(str(checkpoint))


def test_stream_cursor_restore_rejects_out_of_range_cursor():
    stream = D.ByteStream(data=b"abcdef")
    with pytest.raises(ValueError, match="out of range"):
        stream.load_state_dict({"cursor": 999, "path": None, "size": 6})
