"""LRU bounds and exact-FP8 cache parity."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from exlr8.fast_cache import FastExpertCache, RouteTrace, _SlotLRU, cache_bytes


def test_selected_hit_is_not_evicted_for_a_miss():
    cache = _SlotLRU(3, 8)
    initial, _ = cache.select(np.array([[1, 2, 3]], np.uint32))
    current, misses = cache.select(np.array([[4, 1, 5]], np.uint32))
    assert current[0, 1] == initial[0, 0]
    assert set(cache.entries) == {1, 4, 5}
    assert {e for e, _ in misses} == {4, 5}
    assert cache.hits == 1 and cache.misses == 5 and cache.evictions == 2


def test_duplicate_ids_and_lru_order():
    cache = _SlotLRU(3, 8)
    remapped, misses = cache.select(np.array([[2, 1, 2], [1, 3, 3]], np.uint32))
    assert len(misses) == 3
    assert remapped[0, 0] == remapped[0, 2]
    assert remapped[0, 1] == remapped[1, 0]
    cache.select(np.array([1], np.uint32))
    cache.select(np.array([4], np.uint32))
    assert set(cache.entries) == {1, 3, 4}
    assert len(set(cache.entries.values())) == 3


@pytest.mark.parametrize("ids", [[], [1.5], [-1], [8], [0, 1, 2, 3]])
def test_invalid_route_does_not_change_cache(ids):
    cache = _SlotLRU(3, 8)
    cache.select(np.array([5, 6, 7], np.uint32))
    entries = cache.entries.copy()
    with pytest.raises(ValueError):
        cache.select(np.array(ids))
    assert cache.entries == entries
    assert cache.misses == 3


def test_glm_cache_memory_budget():
    assert cache_bytes(6144, 2048, 12) == 454950912
    assert 75 * cache_bytes(6144, 2048, 12) < 32 * 1024**3
    assert 75 * cache_bytes(6144, 2048, 18) < 48 * 1024**3
    assert 75 * cache_bytes(6144, 2048, 19) > 48 * 1024**3


def test_route_trace_is_bounded_and_omits_prefill(tmp_path):
    trace = RouteTrace(token_limit=3, capacities=(3,))
    trace.record(3, np.array([[0, 1], [2, 3]], np.uint32))
    for row in ([7, 8], [0, 1], [1, 2], [2, 0]):
        trace.record(3, np.array([row], np.uint32))
    result = trace.write(tmp_path / "routes.json")
    assert result["layers"]["3"]["routes"] == [[0, 1], [1, 2], [2, 0]]
    assert result["totals"]["3"]["hits"] == 3
    assert result["totals"]["3"]["misses"] == 3
    assert result["totals"]["3"]["all_selected_hit_tokens"] == 1


def test_route_recording_does_not_read_array_values():
    class PendingRoute:
        shape = (1, 9)

        def __array__(self, *args, **kwargs):
            raise AssertionError("recording must not read the GPU route")

    trace = RouteTrace()
    trace.record(3, PendingRoute())
    assert len(trace.records[3]) == 1


metal = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1",
                           reason="set EXLR8_TEST_METAL=1 on a Metal host")


def _fixture(k=4, hidden=256, intermediate=384, experts=11):
    import mlx.core as mx

    rng = np.random.default_rng(503)
    packed, scales = {}, {}
    for projection in ("gate", "up", "down"):
        a, b = (intermediate, hidden) if projection == "down" else (hidden, intermediate)
        packed[projection] = mx.array(rng.integers(0, 256, experts * a * b * k // 8,
                                                  dtype=np.uint8))
        scales[projection] = (
            mx.array(rng.uniform(-0.08, 0.08, (experts, a)).astype(np.float16)),
            mx.array(rng.uniform(-0.5, 0.5, (experts, b)).astype(np.float16)),
        )
    return SimpleNamespace(hidden=hidden, intermediate=intermediate, k=k,
                           expert_bytes=hidden * intermediate * k // 8,
                           packed=packed, scales=scales,
                           store=SimpleNamespace(resident=True))


@metal
@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("per_slot", [False, True])
def test_cache_cold_warm_and_replacement_match_reference(k, dtype_name, per_slot):
    import mlx.core as mx
    from mlx_lm.models.activations import swiglu
    from exlr8.trellis import expert_linear

    layer = _fixture(k)
    cache = FastExpertCache(layer, 4)
    rng = np.random.default_rng(99)
    for values in ([[2, 0], [1, 2]], [[2, 0], [1, 2]], [[10, 3], [0, 10]]):
        ids = mx.array(values, mx.uint32)
        shape = (2, 2, 256) if per_slot else (2, 256)
        x = mx.array(rng.normal(size=shape), dtype=getattr(mx, dtype_name))
        gate = expert_linear(x, ids, layer.packed["gate"], *layer.scales["gate"],
                             k, 256, 384, 1)
        up = expert_linear(x, ids, layer.packed["up"], *layer.scales["up"],
                           k, 256, 384, 1)
        expected = expert_linear(swiglu(gate, up), ids, layer.packed["down"],
                                 *layer.scales["down"], k, 384, 256, 2)
        actual = cache(x, ids)
        assert actual.shape == (2, 2, 256) and actual.dtype == x.dtype
        a, b = np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
        limit = {"float32": 4e-6, "float16": 0.001, "bfloat16": 0.008}[dtype_name]
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < limit
    assert cache.stats()["evictions"] == 1


@metal
@pytest.mark.parametrize("k", [3, 4])
def test_cached_codes_keep_the_decoded_values(k):
    import mlx.core as mx
    from exlr8.trellis import decode_trellis_reference

    layer = _fixture(k)
    cache = FastExpertCache(layer, 3)
    cache.prepare(np.array([[10, 2, 7]], np.uint32))
    for expert, slot in cache._lru.entries.items():
        for projection in ("gate", "up", "down"):
            a, b, side = (384, 256, 2) if projection == "down" else (256, 384, 1)
            packed = np.array(cache._packed(expert, projection))
            expected = decode_trellis_reference(packed, k, a, b, side).T
            codes = cache._down[slot] if side == 2 else cache._gate_up[slot, int(projection == "up")]
            exponents = mx.full((b, a // 32), 127, mx.uint8)
            actual = np.array(mx.dequantize(codes, scales=exponents, mode="mxfp8",
                                             dtype=mx.float16))
            np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


@metal
def test_cache_hits_do_not_expand_or_stack_weights(monkeypatch):
    import mlx.core as mx
    import exlr8.dense_trellis as dense_trellis

    cache = FastExpertCache(_fixture(), 3)
    ids = np.array([[1, 2, 10]], np.uint32)
    cache.prepare(ids)

    def unexpected(*args, **kwargs):
        raise AssertionError("cache hit must use the prepared buffers")

    monkeypatch.setattr(dense_trellis, "expand_trellis", unexpected)
    monkeypatch.setattr(mx, "stack", unexpected)
    output = cache(mx.ones((1, 256), mx.bfloat16), ids)
    assert bool(mx.all(mx.isfinite(output)).item())
