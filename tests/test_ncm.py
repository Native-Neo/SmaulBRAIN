import torch

from ncm_train import ByteBatchStream
from ncm import (
    NCMConfig, NCMGenerator, NCMTransformer, ResidualVQTokenizer,
    SmaulRecurrentBlock, ncm_loss,
)


def tiny_config():
    return NCMConfig(
        vocab_size=256,
        d_model=16,
        codebook_size=16,
        levels=3,
        n_layers=1,
        n_heads=4,
        max_seq_len=16,
        dropout=0.0,
    )


def test_tokenizer_emits_multiscale_codes_and_reconstructs_shape():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    data = torch.randint(0, 256, (2, 9))
    logits, loss, codes = tokenizer(data)

    assert logits.shape == (2, 9, 256)
    assert torch.isfinite(loss)
    assert [x.shape for x in codes] == [(2, 3), (2, 5), (2, 9)]
    assert all(int(x.min()) >= 0 and int(x.max()) < config.codebook_size for x in codes)
    decoded = tokenizer.decode(codes, output_length=9)
    assert decoded.shape == data.shape
    assert int(decoded.min()) >= 0 and int(decoded.max()) < config.vocab_size


def test_ncm_predicts_every_level_in_one_forward_and_backpropagates():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    data = torch.randint(0, 256, (2, 9))
    codes = tokenizer.encode(data)
    model = NCMTransformer(config, tokenizer)

    logits = model(codes)
    assert [x.shape for x in logits] == [(2, 3, 16), (2, 5, 16), (2, 9, 16)]
    loss = ncm_loss(model, codes)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None for p in tokenizer.parameters())


def test_current_level_input_does_not_use_its_own_target_codes():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    model = NCMTransformer(config, tokenizer)
    codes = tokenizer.encode(torch.randint(0, 256, (1, 9)))
    before, lengths = model._build_level_inputs(codes, corrupt=False)
    changed = list(codes)
    changed[1] = torch.randint(0, config.codebook_size, changed[1].shape)
    after, _ = model._build_level_inputs(changed, corrupt=False)
    start = lengths[0]
    end = start + lengths[1]
    assert torch.equal(before[:, start:end], after[:, start:end])


def test_level_prediction_depends_only_on_earlier_codes():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    model = NCMTransformer(config, tokenizer).eval()
    codes = tokenizer.encode(torch.randint(0, 256, (1, 9)))
    original = model.predict_level(codes, 1)
    changed = list(codes)
    changed[1] = torch.randint(0, config.codebook_size, changed[1].shape)
    changed[2] = torch.randint(0, config.codebook_size, changed[2].shape)
    after = model.predict_level(changed, 1)
    assert original.shape == (1, 5, config.codebook_size)
    assert torch.equal(original, after)


def test_generation_advances_by_levels_and_decodes_bytes():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    model = NCMTransformer(config, tokenizer)
    codes = model.generate_codes([3, 5, 9])
    assert [x.shape for x in codes] == [(1, 3), (1, 5), (1, 9)]
    assert all(int(x.min()) >= 0 and int(x.max()) < config.codebook_size for x in codes)
    generated = model.generate(9)
    assert generated.shape == (1, 9)
    assert int(generated.min()) >= 0 and int(generated.max()) < config.vocab_size


def test_prompt_conditioning_has_valid_shape_and_backpropagates():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    model = NCMTransformer(config, tokenizer)
    codes = tokenizer.encode(torch.randint(0, 256, (2, 9)))
    prefix = torch.randint(0, 256, (2, 4))
    logits = model(codes, prefix_tokens=prefix)
    assert [x.shape for x in logits] == [(2, 3, 16), (2, 5, 16), (2, 9, 16)]
    loss = ncm_loss(model, codes, prefix_tokens=prefix)
    loss.backward()
    assert any(p.grad is not None for p in model.prefix_projection.parameters())
    generated = model.generate(9, prefix_tokens=prefix)
    assert generated.shape == (2, 9)


def test_streaming_byte_batcher_reads_text_without_loading_whole_file(tmp_path):
    corpus = tmp_path / "sample.txt"
    corpus.write_bytes(bytes(range(128)) * 8)
    stream = ByteBatchStream(corpus, seq_len=16, batch_size=2, buffer_bytes=64)
    batch = stream.next(torch.device("cpu"))
    assert batch.shape == (2, 16)
    assert batch.dtype == torch.long
    assert int(batch.min()) >= 0 and int(batch.max()) < 256


def test_generator_uses_smaul_recurrent_blocks_instead_of_transformers():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    model = NCMGenerator(config, tokenizer)

    assert all(isinstance(block, SmaulRecurrentBlock) for block in model.blocks)
    assert not any(
        isinstance(module, (torch.nn.TransformerEncoder, torch.nn.TransformerEncoderLayer))
        for module in model.modules()
    )
    # Keep the former public name working for scripts/checkpoints made before this change.
    assert NCMTransformer is NCMGenerator


def test_chunked_recurrent_scan_matches_the_recurrence_equation():
    block = SmaulRecurrentBlock(dimension=3, feedforward_dimension=12)
    values = torch.randn(2, 73, 3, requires_grad=True)
    rate = torch.tensor([0.01, 0.03, 0.08])
    actual = block._scan(values, rate)

    state = torch.zeros(2, 3)
    expected = []
    for position in range(values.shape[1]):
        state = torch.exp(-rate) * state + values[:, position, :]
        expected.append(state)
    expected = torch.stack(expected, dim=1)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_recurrent_config_does_not_require_attention_head_splitting():
    # n_heads is accepted only for backward compatibility; recurrent blocks use
    # the full channel dimension and never split it into Transformer heads.
    config = NCMConfig(d_model=10, n_heads=4)
    model = NCMGenerator(config, ResidualVQTokenizer(config))
    assert model.config.d_model == 10
