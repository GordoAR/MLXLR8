"""Exact static FP8 codes, stored tile order, and GPU-only decode routing."""

import os

import numpy as np
import pytest

from exlr8.static_cache import (StaticExpertCache, _e4m3_codes, _expert_map,
                                codebook_codes, expand_static_reference,
                                static_cache_bytes)
from exlr8.trellis import decode_trellis_reference, mcg_codebook


def _fp8_values(codes):
    magnitude = codes & 127
    exponent, fraction = magnitude >> 3, magnitude & 7
    values = np.where(exponent == 0, fraction * 2.0**-9,
                      (8 + fraction) * np.exp2(exponent.astype(np.int32) - 10))
    return np.copysign(values, np.where(codes & 128, -1, 1)).astype(np.float16)


def test_static_codebook_keeps_every_value_and_signed_zero():
    actual = _fp8_values(codebook_codes())
    np.testing.assert_array_equal(actual.view(np.uint16), mcg_codebook().view(np.uint16))


def test_map_keeps_the_given_slot_order_and_miss_sentinel():
    ids, mapping = _expert_map([256, 17, 3], 257)
    assert ids == (256, 17, 3)
    assert mapping.dtype == np.uint32
    np.testing.assert_array_equal(mapping[[256, 17, 3]], [0, 1, 2])
    assert np.all(mapping[np.setdiff1d(np.arange(257), [256, 17, 3])] == 0xFFFFFFFF)


@pytest.mark.parametrize("ids", [[], [1.5], [-1], [257], [3, 3], [[3]]])
def test_invalid_hot_ids(ids):
    with pytest.raises(ValueError):
        _expert_map(ids, 257)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("side", [1, 2])
def test_static_tile_bytes_match_the_physical_matrix(k, side):
    h, m = 256, 384
    raw = np.random.default_rng(53).integers(0, 256, h * m * k // 8, np.uint8)
    matrix = decode_trellis_reference(raw, k, h, m, side)
    tiles = matrix.reshape(h // 16, 16, m // 16, 16).transpose(0, 2, 1, 3)
    lane, j = np.arange(256) // 8, np.arange(256) % 8
    position = ((lane & 3) * 2 + (j & 1) + (j & 2) * 4) * 16 + (lane >> 2) + (j & 4) * 2
    tiles = tiles.reshape(h // 16, m // 16, 256)[..., position]
    if side == 2:
        tiles = tiles.reshape(h // 128, 8, m // 128, 8, 256).transpose(0, 2, 1, 3, 4)
    else:
        tiles = tiles.reshape(h // 16, m // 128, 8, 256).transpose(1, 0, 2, 3)
    expected = _e4m3_codes(tiles).reshape(-1)
    np.testing.assert_array_equal(expand_static_reference(raw, k, h, m, side), expected)


def test_static_memory_budget():
    assert static_cache_bytes(6144, 2048, 1) == 37749764
    assert 75 * static_cache_bytes(6144, 2048, 18) < 48 * 1024**3
    assert 75 * static_cache_bytes(6144, 2048, 19) > 48 * 1024**3


def test_hot_lists_use_train_routes_only_and_break_ties_by_id():
    from tools.static_cache_routes import fixed_hot_lists

    trace = {"layers": {"3": {"routes": [[8, 4, 256], [4, 8, 256],
                                         [17, 8, 256], [17, 4, 256]]}}}
    result = fixed_hot_lists(trace, split=2, capacities=(1, 2, 3))
    assert result["layers"]["3"]["2"]["expert_ids"] == [256, 4]
    assert result["layers"]["3"]["3"]["expert_ids"] == [256, 4, 8]
    assert result["totals"]["heldout"]["3"]["hit_rate"] == 4 / 6
    assert result["totals"]["heldout"]["3"]["routed_hit_rate"] == 2 / 4
    assert result["totals"]["heldout"]["1"]["all_selected_hit_token_rate"] == 0


def test_hot_lists_require_a_complete_split():
    from tools.static_cache_routes import fixed_hot_lists

    with pytest.raises(ValueError):
        fixed_hot_lists({"layers": {"3": {"routes": [[0, 256]]}}}, split=1)


metal = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1",
                           reason="set EXLR8_TEST_METAL=1 on a Metal host")


@metal
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("side", [1, 2])
def test_metal_static_expansion_matches_all_tile_bytes(k, side):
    import mlx.core as mx
    from exlr8.static_cache import expand_static_trellis

    raw = np.random.default_rng(29).integers(0, 256, 2 * 256 * 384 * k // 8, np.uint8)
    expected = expand_static_reference(raw, k, 256, 384, side)
    actual = expand_static_trellis(mx.array(raw), k, 256, 384, side)
    assert actual.dtype == mx.uint8
    np.testing.assert_array_equal(np.array(actual), expected)


@metal
@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("hot_ids", [[10], [0, 1, 2, 10]])
def test_static_decode_is_bit_equal_with_cached_and_uncached_experts(k, dtype_name, hot_ids):
    import mlx.core as mx
    from exlr8.fast_decode import fast_expert_mlp
    from test_fast_cache import _fixture

    layer = _fixture(k)
    cache = StaticExpertCache(layer, hot_ids)
    ids = mx.array([[2, 0, 10], [1, 7, 10]], mx.uint32)
    x = mx.array(np.random.default_rng(15).normal(size=(2, 256)),
                 dtype=getattr(mx, dtype_name))
    expected = fast_expert_mlp(x, ids, layer.packed, layer.scales, k, 256, 384)
    actual = cache(x, ids)
    a, b = np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


@metal
def test_static_decode_does_not_read_routes_or_change_weight_buffers(monkeypatch):
    import mlx.core as mx
    import exlr8.static_cache as static_cache
    from test_fast_cache import _fixture

    cache = StaticExpertCache(_fixture(), [10])
    x = mx.ones((1, 256), mx.bfloat16)
    ids = mx.array([[1, 2, 10]], mx.uint32)
    mx.eval(cache(x, ids))

    def unexpected(*args, **kwargs):
        raise AssertionError("static decode must use its fixed buffers and GPU map")

    monkeypatch.setattr(static_cache, "expand_static_trellis", unexpected)
    monkeypatch.setattr(np, "asarray", unexpected)
    monkeypatch.setattr(mx, "stack", unexpected)
    output = cache(x, ids)
    assert bool(mx.all(mx.isfinite(output)).item())
