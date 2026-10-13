"""Mixed half/F32 attention checks; GPU execution needs the test gate."""

import os

import numpy as np
import pytest


metal = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1",
                          reason="set EXLR8_TEST_METAL=1 on a Metal host")


def test_bf16_to_half_is_exact_in_supported_range():
    values = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    supported = np.isfinite(values) & (np.abs(values) >= 2.0**-17) & (np.abs(values) <= 65504)
    selected = values[supported]
    np.testing.assert_array_equal(selected.astype(np.float16).astype(np.float32), selected)


def test_half_probability_format_limits():
    values = np.array([2.0**-14, 2.0**-17, 2.0**-24, 2.0**-25, 2.0**-26], dtype=np.float32)
    expected = np.array([2.0**-14, 2.0**-17, 2.0**-24, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(values.astype(np.float16).astype(np.float32), expected)


@metal
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("shape", [(2, 19, 71, 35), (1, 64, 2048, 512), (2, 64, 512, 2048)])
def test_half_gemm_matches_f32_reference(transpose_b, shape):
    import mlx.core as mx
    from exlr8.dsa_prefill_half import half_gemm

    batch, m, n, k = shape
    rng = np.random.default_rng(71)
    a = rng.integers(-16, 17, (batch, m, k)).astype(np.float32) / 16
    b = rng.integers(-16, 17, (batch, n, k) if transpose_b else (batch, k, n)).astype(np.float32) / 16
    expected = a @ (b.swapaxes(-1, -2) if transpose_b else b)
    actual = half_gemm(mx.array(a, dtype=mx.bfloat16), mx.array(b, dtype=mx.bfloat16),
                       transpose_b=transpose_b, output_dtype=mx.float32)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@metal
def test_half_operands_have_f32_accumulators():
    import mlx.core as mx
    from exlr8.dsa_prefill_half import half_gemm

    a = mx.full((1, 8, 8), 4096, dtype=mx.bfloat16)
    actual = half_gemm(a, a, output_dtype=mx.float32)
    np.testing.assert_array_equal(np.asarray(actual), np.full((1, 8, 8), 8 * 4096.0**2, dtype=np.float32))


@metal
def test_half_matrix_probability_underflow_limit():
    import mlx.core as mx
    from exlr8.dsa_prefill_half import half_gemm

    values = np.array([2.0**-14, 2.0**-17, 2.0**-24, 2.0**-25, 2.0**-26, 0, 0, 0], dtype=np.float32)
    a = mx.array(values[None, None, :], dtype=mx.bfloat16)
    b = mx.eye(8, dtype=mx.bfloat16)[None, :, :]
    actual = half_gemm(a, b, output_dtype=mx.float32)
    np.testing.assert_array_equal(np.asarray(actual).reshape(-1), values.astype(np.float16).astype(np.float32))


@metal
@pytest.mark.parametrize("half_qk,half_pv", [(True, False), (False, True), (True, True)])
def test_half_attention_preserves_masks_and_bf16_boundaries(half_qk, half_pv):
    import mlx.core as mx
    from exlr8.dsa_prefill import heads_as_rows_attention
    from exlr8.dsa_prefill_half import half_heads_as_rows_attention
    from test_dsa_prefill import inputs

    q, qp, latent, pe, ids = inputs(batch=2, heads=64, length=3, keys=131,
                                  dims=512, pe=64, selected=97, dtype=mx.bfloat16)
    mask = (mx.arange(131) % 4 != 3)[None, None, None, :]
    expected = heads_as_rows_attention(q, qp, latent, pe, ids, mask, 0.0625)
    actual = half_heads_as_rows_attention(q, qp, latent, pe, ids, mask, 0.0625,
                                          half_qk=half_qk, half_pv=half_pv, chunk=2)
    a, b = [np.asarray(x.astype(mx.float32)) for x in (actual, expected)]
    assert actual.dtype == mx.bfloat16
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 0.002
