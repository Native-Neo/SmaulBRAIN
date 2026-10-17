import numpy as np
import pytest
import torch

from smaulbrain import quant as Q


@pytest.mark.parametrize("shape", [(9, 16), (12, 32), (7, 19), (33, 72)])
def test_tiled_packed_forward_transpose_and_unpack_match_row_major(shape):
    if not Q.prepare_ternary_backend():
        pytest.skip("native C++ backend unavailable")
    rng = np.random.default_rng(2200 + sum(shape))
    din, dout = shape
    weights = rng.standard_normal(shape).astype(np.float32)
    inputs = rng.standard_normal((3, din)).astype(np.float32)
    upstream = rng.standard_normal((3, dout)).astype(np.float32)

    packed, scale, packed_shape, _ = Q.pack_ternary_tiled(torch.from_numpy(weights))
    assert isinstance(packed, Q.TiledPacked)
    codes = Q.unpack_codes(packed, packed_shape)
    row_major, row_scale, _, _ = Q.pack_ternary(weights)
    assert np.array_equal(codes.numpy(), Q.unpack_codes(row_major, packed_shape).numpy())
    assert np.isclose(scale, row_scale)

    dense = codes.float() * scale
    expected = torch.from_numpy(inputs) @ dense
    expected_dx = torch.from_numpy(upstream) @ dense.T
    actual = Q.packed_matmat(torch.from_numpy(inputs), packed, packed_shape, scale)
    actual_dx = Q.packed_matmat_transpose(
        torch.from_numpy(upstream), packed, packed_shape, scale
    )
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert torch.allclose(actual_dx, expected_dx, atol=3e-5, rtol=3e-5)
