import pytest
import torch

import ncm
from ncm_train import ByteBatchStream, SmaulTrainingOptimizer, build_ncm_config
from ncm import (
    NCMConfig, NCMGenerator, NCMLinear, NCMTransformer, ResidualVQTokenizer,
    SmaulDenseFFN, SmaulRecurrentBlock, SmaulSharedMoE,
    apply_direct_ternary_updates, migrate_ternary_state_dict, ncm_loss,
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


def test_ncm_training_optimizer_uses_smaulopt_and_restores_state():
    from smaulbrain.smaulopt import SmaulOPT

    model = torch.nn.Linear(4, 3)
    optimizer = SmaulTrainingOptimizer(model, lr=1e-3)
    assert isinstance(optimizer.optimizer, SmaulOPT)

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss = model(torch.randn(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())

    checkpoint = optimizer.state_dict()
    restored = SmaulTrainingOptimizer(model, lr=9e-4)
    restored.load_state_dict(checkpoint)
    assert restored.optimizer.lr == 1e-3
    assert restored.optimizer.n_updates == 1
    assert restored.optimizer.state.keys() == optimizer.optimizer.state.keys()


def test_ncm_smaulopt_resumes_legacy_adamw_checkpoint_with_fresh_state(capsys):
    model = torch.nn.Linear(4, 3)
    legacy_optimizer = torch.optim.AdamW(model.parameters())
    legacy_state = legacy_optimizer.state_dict()

    optimizer = SmaulTrainingOptimizer(model, lr=1e-3)
    optimizer.load_state_dict(legacy_state)

    assert optimizer.optimizer.n_updates == 0
    assert optimizer.optimizer.state
    assert "checkpoint uses AdamW" in capsys.readouterr().err


def test_tokenizer_emits_multiscale_codes_and_reconstructs_shape():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    data = torch.randint(0, 256, (2, 9))
    logits, loss, codes = tokenizer(data)

    assert logits.shape == (2, 9, 256)
    assert torch.isfinite(loss)
    assert set(tokenizer.last_loss_components) == {"reconstruction", "quantization", "total"}
    assert all(torch.isfinite(value) for value in tokenizer.last_loss_components.values())
    assert torch.allclose(tokenizer.last_loss_components["total"], loss.detach())
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


def test_generator_gradient_checkpointing_matches_plain_backward():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    codes = tokenizer.encode(torch.randint(0, 256, (2, 9)))
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    def run(checkpointing):
        torch.manual_seed(29)
        model = NCMGenerator(
            config, tokenizer, gradient_checkpointing=checkpointing
        )
        model.train()
        torch.manual_seed(31)
        loss = ncm_loss(model, codes)
        loss.backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), grads

    plain_loss, plain_grads = run(False)
    checkpointed_loss, checkpointed_grads = run(True)

    assert torch.allclose(plain_loss, checkpointed_loss)
    assert plain_grads.keys() == checkpointed_grads.keys()
    for name, plain in plain_grads.items():
        assert torch.allclose(plain, checkpointed_grads[name], atol=1e-6, rtol=1e-5), name


def test_iter_level_logits_matches_the_joint_forward():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    codes = tokenizer.encode(torch.randint(0, 256, (2, 9)))
    prefix = torch.randint(0, 256, (2, 5))
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    torch.manual_seed(37)
    model = NCMGenerator(config, tokenizer)
    model.eval()
    joint = model(codes, prefix_tokens=prefix, corrupt=False)
    per_level = [
        logits for logits, _ in
        model.iter_level_logits(codes, prefix_tokens=prefix, corrupt=False)
    ]
    assert len(per_level) == len(joint)
    for chunked, whole in zip(per_level, joint):
        assert torch.allclose(chunked, whole, atol=1e-6, rtol=1e-6)


def test_chunked_loss_backward_matches_the_joint_loss():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    codes = tokenizer.encode(torch.randint(0, 256, (2, 9)))
    prefix = torch.randint(0, 256, (2, 5))
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    def run(chunked):
        torch.manual_seed(41)
        model = NCMGenerator(config, tokenizer)
        model.train()
        torch.manual_seed(43)
        model.zero_grad(set_to_none=True)
        if chunked:
            loss = ncm.backward_ncm_loss_chunked(
                model, codes, prefix_tokens=prefix
            )
        else:
            loss = ncm_loss(model, codes, prefix_tokens=prefix)
            loss.backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), grads

    joint_loss, joint_grads = run(False)
    chunked_loss, chunked_grads = run(True)
    assert torch.allclose(joint_loss, chunked_loss)
    assert joint_grads.keys() == chunked_grads.keys()
    for name, joint in joint_grads.items():
        assert torch.allclose(joint, chunked_grads[name], atol=1e-6, rtol=1e-5), name


def test_chunked_recurrent_scan_matches_the_recurrence_equation(monkeypatch):
    monkeypatch.setattr(ncm._quant, "prepare_ternary_backend", lambda: False)
    block = SmaulRecurrentBlock(dimension=3)
    values = torch.randn(2, 173, 3, requires_grad=True)
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



def test_grouped_small_moe_matches_sparse_routing_reference():
    torch.manual_seed(7)
    moe = SmaulSharedMoE(dimension=8, n_experts=4, top_k=2).eval()
    inputs = torch.randn(2, 7, 8)
    actual = moe(inputs)

    flat = inputs.reshape(-1, 8)
    route_logits = moe.router(flat)
    chosen_logits, chosen_experts = route_logits.topk(2, dim=-1)
    route_weights = torch.softmax(chosen_logits, dim=-1)
    expected = torch.zeros_like(flat)
    for expert_id, expert in enumerate(moe.experts):
        token_indices, slots = torch.where(chosen_experts == expert_id)
        if token_indices.numel() == 0:
            continue
        expert_output = expert(flat.index_select(0, token_indices))
        weights = route_weights[token_indices, slots].unsqueeze(-1)
        expected = expected.index_add(0, token_indices, expert_output * weights)

    assert torch.allclose(actual, expected.reshape_as(inputs), atol=1e-6, rtol=1e-6)


def test_shared_moe_pool_routes_tokens_and_receives_gradients():
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    model = NCMGenerator(config, tokenizer)

    assert isinstance(model.channel_mixer, SmaulSharedMoE)
    assert sum(isinstance(module, SmaulSharedMoE) for module in model.modules()) == 1
    assert model.channel_mixer.n_experts == config.n_experts
    assert model.channel_mixer.top_k == config.top_k

    codes = tokenizer.encode(torch.randint(0, 256, (2, 9)))
    loss = ncm_loss(model, codes)
    loss.backward()

    router_gradient = model.channel_mixer.router.weight.grad
    assert router_gradient is not None
    assert torch.isfinite(router_gradient).all()
    assert float(router_gradient.abs().sum()) > 0.0


def _naive_scan(values, rate):
    decay = torch.exp(-rate)
    state = torch.zeros_like(values[:, 0])
    out = []
    for step in range(values.shape[1]):
        state = decay * state + values[:, step]
        out.append(state)
    return torch.stack(out, dim=1)


def test_native_recurrent_scan_backward_matches_autograd_reference():
    if not ncm._quant.prepare_ternary_backend():
        pytest.skip("native recurrent scan backend is unavailable")
    torch.manual_seed(23)
    block = SmaulRecurrentBlock(dimension=3)
    values = torch.randn(2, 173, 3, requires_grad=True)
    rate = torch.tensor([0.01, 0.03, 0.08], requires_grad=True)
    reference_values = values.detach().clone().requires_grad_(True)
    reference_rate = rate.detach().clone().requires_grad_(True)
    weights = torch.randn_like(values)

    actual = block._scan(values, rate)
    state = torch.zeros(2, 3)
    expected_parts = []
    alpha = torch.exp(-reference_rate)
    for position in range(reference_values.shape[1]):
        state = alpha * state + reference_values[:, position, :]
        expected_parts.append(state)
    expected = torch.stack(expected_parts, dim=1)

    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(values.grad, reference_values.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(rate.grad, reference_rate.grad, atol=2e-5, rtol=2e-5)


def test_block_bidirectional_scan_matches_naive_recurrence_across_chunks():
    torch.manual_seed(3)
    block = SmaulRecurrentBlock(8).eval()
    moe = SmaulSharedMoE(dimension=8, n_experts=4, top_k=2).eval()
    # 300 > _SCAN_CHUNK exercises chunk carry-over; 5 checks segment reset.
    lengths = [5, 300]
    x = torch.randn(2, sum(lengths), 8)
    actual = block(x, lengths, moe)

    with torch.no_grad():
        normalized = block.norm_attention(x)
        gate = torch.sigmoid(block.W_r(normalized))
        key_value = block.W_k(normalized) * block.W_v(normalized)
        rate = torch.nn.functional.softplus(block.decay_logit).clamp(min=1e-4, max=0.1)
        contexts, offset = [], 0
        for length in lengths:
            seg = key_value[:, offset:offset + length]
            fwd = _naive_scan(seg, rate)
            rev = _naive_scan(seg.flip(1), rate).flip(1)
            contexts.append(0.5 * (fwd + rev))
            offset += length
        context = torch.cat(contexts, dim=1)
        hidden = x + block.W_o(gate * (context + block.bonus * key_value))
        expected = hidden + moe(block.norm_channel(hidden))

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_all_four_presets_map_to_ncm_architectures():
    expected = {
        "dense-tiny": ("dense", 32, 15, 128, 0, 0),
        "dense-full": ("dense", 1000, 19, 3152, 0, 0),
        "moe-tiny": ("moe", 64, 4, 0, 32, 2),
        "moe-full": ("moe", 512, 6, 0, 64, 8),
    }
    for preset, values in expected.items():
        config = build_ncm_config(preset, seq_len=128)
        actual = (
            config.model_kind, config.d_model, config.n_layers,
            config.ffn_hidden if config.model_kind == "dense" else 0,
            config.n_experts, config.top_k,
        )
        assert actual == values
        assert config.preset == preset


def test_dense_preset_uses_per_layer_swiglu_and_backpropagates():
    config = build_ncm_config(
        "dense-tiny", seq_len=16, d_model=8, layers=1,
        levels=3, codebook_size=16,
    )
    tokenizer = ResidualVQTokenizer(config)
    model = NCMGenerator(config, tokenizer)
    assert model.channel_mixer is None
    assert len(model.dense_mixers) == config.n_layers
    assert isinstance(model.dense_mixers[0], SmaulDenseFFN)
    codes = tokenizer.encode(torch.randint(0, 256, (2, 9)))
    loss = ncm_loss(model, codes)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.dense_mixers[0].W_a.weight.grad is not None


def test_ncm_linear_ternary_forward_and_backward_use_packed_weights():
    torch.manual_seed(11)
    layer = NCMLinear(
        5, 4, forward_quant="ternary", backward_mode="ternary"
    )
    layer.ternary_update_rate = 0.5
    inputs = torch.randn(2, 3, 5, requires_grad=True)
    packed_before = layer.ternary_packed.clone()
    output = layer(inputs)
    output.square().mean().backward()
    apply_direct_ternary_updates(layer)

    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert "weight" not in dict(layer.named_parameters())
    assert layer.ternary_packed.dtype == torch.uint8
    assert layer.ternary_packed.numel() == 5  # 20 ternary weights, four codes per byte.
    assert not torch.equal(layer.ternary_packed, packed_before)
    assert int(layer.ternary_updates) > 0
    assert layer._cached_fp8_weight is None


def test_ternary_compute_layout_matches_canonical_packed_codes():
    """The fast compute layout must not change the packed-weight math."""
    from smaulbrain import quant as _quant

    torch.manual_seed(17)
    layer = NCMLinear(
        32, 24, forward_quant="ternary", backward_mode="ternary"
    )
    inputs = torch.randn(4, 32)
    scale = float(layer.ternary_scale.item())
    canonical = _quant.packed_matmat(
        inputs, layer.ternary_packed, layer._ternary_shape, scale
    )
    compute = layer._packed_for_compute()
    fast = _quant.packed_matmat(inputs, compute, layer._ternary_shape, scale)
    assert torch.allclose(canonical, fast, atol=1e-6, rtol=1e-6)
    # The canonical buffer stays the checkpoint format (row-major 2-bit codes).
    assert layer.ternary_packed.numel() == (32 * 24 + 3) // 4


def test_ternary_compute_layout_is_rebuilt_after_sparse_updates():
    from smaulbrain import quant as _quant

    torch.manual_seed(19)
    layer = NCMLinear(
        32, 24, forward_quant="ternary", backward_mode="ternary"
    )
    layer.ternary_update_rate = 1.0
    inputs = torch.randn(4, 32)
    scale = float(layer.ternary_scale.item())
    layer._record_ternary_update(torch.randn(24, 32))
    apply_direct_ternary_updates(layer)
    assert int(layer.ternary_updates) > 0

    canonical = _quant.packed_matmat(
        inputs, layer.ternary_packed, layer._ternary_shape, scale
    )
    fast = _quant.packed_matmat(
        inputs, layer._packed_for_compute(), layer._ternary_shape, scale
    )
    assert torch.allclose(canonical, fast, atol=1e-6, rtol=1e-6)


def test_tokenizer_encode_skips_the_reconstruction_head():
    """encode() must not run the decoder: it only needs the code ids."""
    config = tiny_config()
    tokenizer = ResidualVQTokenizer(config)
    data = torch.randint(0, 256, (2, 9))
    expected = tokenizer.encode(data)

    calls = []
    original = tokenizer.decoder.forward

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    tokenizer.decoder.forward = spy
    codes = tokenizer.encode(data)
    assert calls == []
    assert len(codes) == len(expected)
    for got, want in zip(codes, expected):
        assert torch.equal(got, want)


def test_legacy_fp32_master_ternary_checkpoint_migrates_to_packed_codes():
    layer = NCMLinear(5, 4, forward_quant="ternary", backward_mode="ternary")
    legacy_state = {
        "weight": torch.randn(4, 5),
        "bias": torch.randn(4),
    }

    converted, migrated = migrate_ternary_state_dict(layer, legacy_state)

    assert migrated
    assert "weight" not in converted
    assert converted["ternary_packed"].dtype == torch.uint8
    assert converted["ternary_packed"].numel() == 5
    assert float(converted["ternary_scale"]) > 0
    layer.load_state_dict(converted)
    assert "weight" not in dict(layer.named_parameters())


def test_tokenizer_ternary_qat_has_gradients_for_trainable_parameters():
    config = NCMConfig(
        d_model=8, codebook_size=8, levels=2, max_seq_len=16,
        forward_quant="ternary", backward_mode="ternary",
    )
    tokenizer = ResidualVQTokenizer(config)
    inputs = torch.randint(0, 256, (2, 9))
    _, loss, _ = tokenizer(inputs)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in tokenizer.named_parameters()
        if isinstance(dict(tokenizer.named_modules()).get(name.rsplit(".", 1)[0]), NCMLinear)
    )



def test_ternary_update_gradient_is_computed_in_global_mean_chunks(monkeypatch):
    monkeypatch.setattr(ncm, "_TERNARY_GRADIENT_CHUNK_COLUMNS", 2)
    sampled_chunks = []

    def fake_chunk_sampler(
        gradient, rate, seed, global_mean_abs, full_cols, column_offset
    ):
        sampled_chunks.append(
            (gradient.detach().clone(), rate, seed, global_mean_abs, full_cols, column_offset)
        )
        return (
            torch.tensor([column_offset], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int8),
        )

    monkeypatch.setattr(
        ncm._quant, "sample_ternary_gradient_update_chunk",
        fake_chunk_sampler, raising=False,
    )
    layer = NCMLinear(
        3, 5, forward_quant="ternary", backward_mode="ternary"
    )
    layer.ternary_update_rate = 0.2
    torch.manual_seed(101)
    dy = torch.randn(4, 5)
    x_flat = torch.randn(4, 3)

    layer._record_ternary_update_from_inputs(dy, x_flat)

    full_gradient = (dy.t() @ x_flat).t()
    expected_mean = float(
        full_gradient.abs().sum(dtype=torch.float64).item()
    ) / full_gradient.numel()
    assert [chunk[5] for chunk in sampled_chunks] == [0, 2, 4]
    assert [chunk[4] for chunk in sampled_chunks] == [5, 5, 5]
    assert torch.allclose(
        torch.cat([chunk[0] for chunk in sampled_chunks], dim=1),
        full_gradient,
    )
    assert all(abs(chunk[3] - expected_mean) < 1e-7 for chunk in sampled_chunks)
    assert [segment.tolist() for segment in layer._pending_ternary_indices] == [
        [0], [2], [4]
    ]
    assert all(segment.tolist() == [-1] for segment in layer._pending_ternary_directions)
    assert "weight" not in dict(layer.named_parameters())


def test_failed_native_ternary_update_clears_queued_segments(monkeypatch):
    layer = NCMLinear(
        3, 5, forward_quant="ternary", backward_mode="ternary"
    )
    layer._pending_ternary_indices.append(torch.tensor([0], dtype=torch.int32))
    layer._pending_ternary_directions.append(torch.tensor([1], dtype=torch.int8))

    def fail_applier(*args, **kwargs):
        raise RuntimeError("simulated native update failure")

    monkeypatch.setattr(ncm._quant, "apply_ternary_update_segments", fail_applier)
    with pytest.raises(RuntimeError, match="simulated native update failure"):
        layer._apply_pending_ternary_updates()

    assert layer._pending_ternary_indices == []
    assert layer._pending_ternary_directions == []
