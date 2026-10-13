import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from exlr8.dsa_prefill import full_latent_attention, heads_as_rows_attention, split_heads_attention
from exlr8.model import Attention


metal = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1", reason="set EXLR8_TEST_METAL=1 on a Metal host")


def inputs(batch=2, heads=4, length=5, keys=17, dims=32, pe=8, selected=6,
           dtype=mx.float32):
    rng = np.random.default_rng(51)
    def random(shape):
        return mx.array(rng.normal(0, 0.7, shape).astype(np.float32), dtype=dtype)
    ids = np.stack([np.stack([
        np.concatenate(([0], rng.choice(np.arange(1, keys), selected - 1, replace=False)))
        for _ in range(length)
    ]) for _ in range(batch)])[:, None, :, :].astype(np.int32)
    return (random((batch, heads, length, dims)), random((batch, heads, length, pe)),
            random((batch, 1, keys, dims)), random((batch, 1, keys, pe)), mx.array(ids))


def reference(q, qp, latent, pe, ids, mask, scale):
    q, qp, latent, pe = [np.asarray(x.astype(mx.float32)) for x in (q, qp, latent, pe)]
    batch, _, length, _ = q.shape
    keys = latent.shape[2]
    allowed = np.broadcast_to(np.arange(keys)[None, :] <= np.arange(keys - length, keys)[:, None],
                              (batch, 1, length, keys)).copy()
    if ids is not None:
        selected = np.zeros_like(allowed)
        np.put_along_axis(selected, np.asarray(ids), True, axis=-1)
        allowed &= selected
    if mask is not None:
        allowed &= np.asarray(mask)
    scores = (q * scale) @ latent.swapaxes(-1, -2) + (qp * scale) @ pe.swapaxes(-1, -2)
    scores = np.where(allowed, scores, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities @ latent


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("external_mask", [False, True])
def test_full_latent_selection_and_causal_masks(sparse, external_mask):
    q, qp, latent, pe, ids = inputs()
    ids = ids if sparse else None
    mask = (mx.arange(17) % 4 != 3)[None, None, None, :] if external_mask else None
    expected = reference(q, qp, latent, pe, ids, mask, 0.125)
    actual = full_latent_attention(q, qp, latent, pe, ids, mask, 0.125, chunk=2)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-5, atol=2e-6)


def test_split_heads_matches_sparse_reference():
    q, qp, latent, pe, ids = inputs(heads=6)
    mask = (mx.arange(17) % 4 != 3)[None, None, None, :]
    actual = split_heads_attention(q, qp, latent, pe, ids, mask, 0.125, chunk=2, head_chunk=3)
    expected = reference(q, qp, latent, pe, ids, mask, 0.125)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-5, atol=2e-6)


def test_heads_as_rows_matches_sparse_reference():
    q, qp, latent, pe, ids = inputs(heads=6)
    mask = (mx.arange(17) % 4 != 3)[None, None, None, :]
    actual = heads_as_rows_attention(q, qp, latent, pe, ids, mask, 0.125, chunk=2)
    expected = reference(q, qp, latent, pe, ids, mask, 0.125)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-5, atol=2e-6)


@metal
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_d512_vector_split_matches_gathered_path(monkeypatch, dtype):
    q, qp, latent, pe, ids = inputs(batch=1, heads=64, length=2, keys=96,
                                  dims=512, pe=64, selected=64, dtype=dtype)
    monkeypatch.delenv("MLX_SDPA_D512_MIN_KL", raising=False)
    with pytest.raises(ValueError, match="GQA factor 8"):
        split_heads_attention(q, qp, latent, pe, ids, None, 0.0625, force_fused=True)
    monkeypatch.setenv("MLX_SDPA_D512_MIN_KL", "0")
    expected = split_heads_attention(q, qp, latent, pe, ids, None, 0.0625, head_chunk=64)
    actual = split_heads_attention(q, qp, latent, pe, ids, None, 0.0625, force_fused=True)
    a, b = [np.asarray(x.astype(mx.float32)) for x in (actual, expected)]
    error = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert error < (3e-5 if dtype == mx.float32 else 0.015)
    assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_d512_heads_as_rows_matches_gathered_path(dtype):
    q, qp, latent, pe, ids = inputs(batch=1, heads=64, length=4, keys=128,
                                  dims=512, pe=64, selected=64, dtype=dtype)
    expected = split_heads_attention(q, qp, latent, pe, ids, None, 0.0625, head_chunk=64)
    actual = heads_as_rows_attention(q, qp, latent, pe, ids, None, 0.0625)
    a, b = [np.asarray(x.astype(mx.float32)) for x in (actual, expected)]
    error = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert error < (3e-5 if dtype == mx.float32 else 0.005)
    assert actual.dtype == dtype
    model_output = Attention._sparse_attention(SimpleNamespace(scale=0.0625),
                                             q, qp, latent, pe, ids, None)
    np.testing.assert_array_equal(np.asarray(model_output.astype(mx.float32)), a)
