import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from exlr8 import indexer_scores
from exlr8.model import Indexer

pytestmark = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1",
                                reason="set EXLR8_TEST_METAL=1 on a Metal host")


def inputs(rows, keys, *, strided=False):
    mx.random.seed(3600 + rows + keys)
    factor = 2 if strided else 1
    q = mx.random.normal((1, 32, rows, 128 * factor)).astype(mx.bfloat16)[..., ::factor]
    k = mx.random.normal((1, 1, keys, 128 * factor)).astype(mx.bfloat16)[..., ::factor]
    weights = mx.random.normal((1, 32, rows)).astype(mx.float32) * 32**-0.5
    return q, k, weights


def reference(q, k, weights, mask):
    q, k, weights = [np.asarray(value.astype(mx.float32)) for value in (q, k, weights)]
    scores = np.maximum((q @ k.swapaxes(-1, -2)) * np.float32(128**-0.5), 0)
    scores = (scores * weights[..., None]).sum(axis=1, keepdims=True)
    return scores if mask is None else np.where(np.asarray(mask), scores, -np.inf)


@pytest.mark.parametrize("rows,keys", [(1, 1), (2, 63), (3, 64), (4, 65), (1, 2049), (2, 2051), (3, 4096), (4, 4099)])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "sparse"])
def test_independent_scores_and_key_tails(rows, keys, mask_kind):
    q, k, weights = inputs(rows, keys)
    mask = None
    if mask_kind == "causal":
        mask = mx.arange(keys)[None, None, None] <= mx.arange(keys-rows, keys)[None, None, :, None]
    elif mask_kind == "sparse":
        mask = (mx.arange(keys) % 3 != 0)[None, None, None]
    actual = np.asarray(indexer_scores.score(q, k, weights, mask))
    expected = reference(q, k, weights, mask)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_noncontiguous_operands(rows):
    q, k, weights = inputs(rows, 2053, strided=True)
    actual = np.asarray(indexer_scores.score(q, k, weights, None))
    np.testing.assert_allclose(actual, reference(q, k, weights, None), rtol=1e-4, atol=1e-4)


def test_zero_weights_masks_and_native_tie_order():
    q, k, weights = inputs(4, 2051)
    weights = mx.zeros_like(weights)
    mask = mx.ones((4, 2051), mx.bool_)
    mask = mx.where(mx.arange(4)[:, None] == 2, False, mask)
    scores = indexer_scores.score(q, k, weights, mask)
    values = np.asarray(scores)[0, 0]
    assert np.all(values[[0, 1, 3]] == 0)
    assert np.all(np.isneginf(values[2]))
    chosen = mx.argpartition(-scores, 2047, axis=-1)[..., :2048]
    np.testing.assert_array_equal(np.asarray(chosen), np.broadcast_to(np.arange(2048), (1, 1, 4, 2048)))


def test_context_growth_uses_one_kernel_wrapper():
    indexer_scores._kernel.cache_clear()
    for keys in (2049, 2050, 2053):
        q, k, weights = inputs(4, keys)
        mx.eval(indexer_scores.score(q, k, weights, None))
    info = indexer_scores._kernel.cache_info()
    assert info.misses == 1 and info.hits == 2


def make_indexer(rows, keys, *, enabled=True, heads=32, dims=128, dtype=mx.bfloat16):
    mx.random.seed(3636)
    q = mx.random.normal((1, heads, rows, dims)).astype(dtype)
    k = mx.random.normal((1, 1, keys, dims)).astype(dtype)
    weights = mx.random.normal((1, rows, heads)).astype(mx.float32)
    instance = SimpleNamespace(
        args=SimpleNamespace(index_topk=2048, index_n_heads=heads, index_head_dim=dims),
        _fused_scores=enabled,
        wk=lambda x: mx.zeros((1, rows, dims), dtype),
        norm_weight=mx.ones((dims,), dtype), norm_bias=mx.zeros((dims,), dtype),
        rope=lambda value, offset: value,
        wq_b=lambda qr: q.swapaxes(1, 2).reshape(1, rows, heads*dims),
        weights_proj=lambda x: weights,
    )
    cache = SimpleNamespace(offset=keys-rows, update_and_fetch=lambda new, values: (k, None))
    return instance, cache


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_indexer_dispatch_and_key_selection(rows, monkeypatch):
    instance, cache = make_indexer(rows, 2051)
    original = indexer_scores.score
    calls = []

    def observed(q, k, weights, mask):
        calls.append(q.shape[2])
        return original(q, k, weights, mask)

    monkeypatch.setattr(indexer_scores, "score", observed)
    mask = mx.arange(2051)[None, None, None] <= mx.arange(2051-rows, 2051)[None, None, :, None]
    x = mx.zeros((1, rows, 8), mx.bfloat16)
    changed = Indexer.__call__(instance, x, x, mask, cache)
    instance._fused_scores = False
    native = Indexer.__call__(instance, x, x, mask, cache)
    assert calls == [rows]
    np.testing.assert_array_equal(np.asarray(mx.sort(changed)), np.asarray(mx.sort(native)))


@pytest.mark.parametrize("options,rows,keys,mask_dtype", [
    ({"enabled": False}, 4, 2051, mx.bool_),
    ({}, 5, 2051, mx.bool_),
    ({}, 32, 2051, mx.bool_),
    ({}, 4, 2048, mx.bool_),
    ({"heads": 2}, 4, 2051, mx.bool_),
    ({"dims": 64}, 4, 2051, mx.bool_),
    ({"dtype": mx.float32}, 4, 2051, mx.bool_),
    ({}, 4, 2051, mx.float32),
])
def test_native_dispatch_bounds(options, rows, keys, mask_dtype, monkeypatch):
    instance, cache = make_indexer(rows, keys, **options)

    def forbidden(*args):
        raise AssertionError("fused scores must not run")

    monkeypatch.setattr(indexer_scores, "score", forbidden)
    x = mx.zeros((1, rows, 8), mx.bfloat16)
    result = Indexer.__call__(instance, x, x, mx.ones((rows, keys), mask_dtype), cache)
    if keys <= 2048:
        assert result is None
    else:
        mx.eval(result)
        assert result.shape == (1, 1, rows, 2048)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_native_head_reduction_order(rows):
    q, k, weights = inputs(rows, 2051)
    # One active dot dimension isolates the head sum from GEMM reduction order.
    q = mx.concatenate((q[..., :1], mx.zeros_like(q[..., 1:])), axis=-1)
    k = mx.concatenate((k[..., :1], mx.zeros_like(k[..., 1:])), axis=-1)
    raw = q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)
    expected = (mx.maximum(raw * 128**-0.5, 0) * weights[..., None]).sum(axis=1, keepdims=True)
    actual = indexer_scores.score(q, k, weights, None)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
