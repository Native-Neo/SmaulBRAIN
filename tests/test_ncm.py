import torch

from ncm import NCMConfig, NCMTransformer, ResidualVQTokenizer, ncm_loss


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
    model = NCMTransformer(config)

    logits = model(codes)
    assert [x.shape for x in logits] == [(2, 3, 16), (2, 5, 16), (2, 9, 16)]
    loss = ncm_loss(model, codes)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_generation_advances_by_levels_not_individual_positions():
    config = tiny_config()
    model = NCMTransformer(config)
    codes = model.generate_codes([3, 5, 9])
    assert [x.shape for x in codes] == [(1, 3), (1, 5), (1, 9)]
    assert all(int(x.min()) >= 0 and int(x.max()) < config.codebook_size for x in codes)


def test_config_rejects_invalid_attention_dimensions():
    try:
        NCMConfig(d_model=10, n_heads=4)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("invalid attention dimensions were accepted")
