import json

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file
from mlx_lm.models.activations import swiglu

from exlr8.experts import ExpertStore
from exlr8.trellis import trellis_linear


@pytest.fixture
def expert_checkpoint(tmp_path):
    directory = tmp_path / "experts"
    directory.mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    plan = {
        "gate": [[4]] * 256,
        "up": [[4]] * 256,
        "down": [4] * 256,
        "shared": {"gate": [4], "up": [4], "down": 4},
    }
    (manifests / "exlr8-manifest-up1.json").write_text(json.dumps({
        "name": "up1", "layers": {"3": plan},
    }))
    rng = np.random.default_rng(17)
    raw = {}
    index = {
        "signs.in": rng.choice([-1, 1], 128).astype(np.int8),
        "signs.out": rng.choice([-1, 1], 128).astype(np.int8),
    }
    for projection in ("gate", "up", "down"):
        raw["trellis." + projection] = rng.integers(0, 256, 256 * 8192, dtype=np.uint8)
        raw["shared.trellis." + projection] = rng.integers(0, 256, 8192, dtype=np.uint8)
        block_name = "sv_block" if projection == "down" else "su_block"
        channel_name = "suh" if projection == "down" else "svh"
        for prefix, count in (("k4", 256), ("shared.k4", 1)):
            index[f"{prefix}.{projection}.{block_name}"] = (
                rng.uniform(0.01, 0.02, (count, 1)).astype(np.float16)
            )
            index[f"{prefix}.{projection}.{channel_name}"] = (
                rng.uniform(-1, 1, (count, 128)).astype(np.float16)
            )
    save_file(raw, str(directory / "exlr8-layer-003.k4.safetensors"))
    save_file(index, str(directory / "exlr8-layer-003.index.safetensors"))
    return tmp_path, raw


@pytest.mark.parametrize("optimize", [False, True])
def test_compact_routing_shared_expert_and_side_scales(expert_checkpoint, optimize):
    path, _ = expert_checkpoint
    store = ExpertStore(path, cache_gib=1, optimize=optimize)
    layer = store.layer(3)
    x = mx.array(np.random.default_rng(20).normal(size=(2, 128)), dtype=mx.float32)
    ids = mx.array([[256, 9, 2], [2, 9, 2]], dtype=mx.uint32)
    actual = layer(x, ids)
    expected = []
    for row, row_ids in enumerate(ids.tolist()):
        slots = []
        for expert in row_ids:
            weights = store._get(layer, expert)
            gate_up = []
            for p, projection in enumerate(("gate", "up")):
                a, b = layer.scales[projection]
                gate_up.append(trellis_linear(
                    x[row:row + 1], weights[p], mx.array(a[expert]), mx.array(b[expert]),
                    4, 128, 128, sharded_side=1,
                ))
            h = swiglu(gate_up[0], gate_up[1])
            a, b = layer.scales["down"]
            slots.append(trellis_linear(
                h, weights[2], mx.array(a[expert]), mx.array(b[expert]),
                4, 128, 128, sharded_side=2,
            )[0])
        expected.append(mx.stack(slots))
    np.testing.assert_allclose(np.array(actual), np.array(mx.stack(expected)), rtol=2e-4, atol=1e-6)


def test_cache_limit_and_exact_expert_range(expert_checkpoint):
    path, raw = expert_checkpoint
    size = 3 * 8192
    store = ExpertStore(path, cache_gib=size * 2 / 1024**3)
    layer = store.layer(3)
    for expert in (4, 8, 4, 12):
        values = store._get(layer, expert)
        np.testing.assert_array_equal(np.array(values[0]), raw["trellis.gate"][expert * 8192:(expert + 1) * 8192])
    assert set(store.cache) == {(3, 4), (3, 12)}
    assert store.cache_bytes == 2 * size
    assert store.stats()["hits"] == 1
    assert store.stats()["misses"] == 3
    shared = store._get(layer, 256)
    np.testing.assert_array_equal(np.array(shared[0]), raw["shared.trellis.gate"])


def test_resident_weights_scales_and_two_token_parity(expert_checkpoint):
    path, raw = expert_checkpoint
    bounded = ExpertStore(path, cache_gib=1)
    resident = ExpertStore(path, resident=True)
    layer = resident.layer(3)
    x = mx.array(np.random.default_rng(30).normal(size=(2, 128)), dtype=mx.bfloat16)
    ids = mx.array([[256, 9, 2], [2, 255, 256]], dtype=mx.uint32)
    expected = bounded.layer(3)(x, ids)
    actual = layer(x, ids)
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)),
    )
    for projection in ("gate", "up", "down"):
        np.testing.assert_array_equal(
            np.array(layer.packed[projection]),
            np.concatenate([raw["trellis." + projection], raw["shared.trellis." + projection]]),
        )
        for actual_scale, expected_scale in zip(layer.scales[projection], bounded.layer(3).scales[projection]):
            np.testing.assert_array_equal(np.array(actual_scale), expected_scale)
    assert resident.stats()["resident_bytes"] == 3 * 257 * (8192 + 2 * 128 * 2)
    assert resident.cache_bytes == resident.hits == resident.misses == 0
    assert resident.layer(3) is layer


def test_resident_forward_does_not_read_or_copy_weights(expert_checkpoint, monkeypatch):
    path, _ = expert_checkpoint
    store = ExpertStore(path, resident=True)
    layer = store.layer(3)

    def unexpected(*args, **kwargs):
        raise AssertionError("resident forward must use its prepared GPU buffers")

    monkeypatch.setattr(store, "_get", unexpected)
    monkeypatch.setattr(layer.weights, "load", unexpected)
    monkeypatch.setattr(np, "unique", unexpected)
    monkeypatch.setattr(mx, "stack", unexpected)
    x = mx.ones((2, 128), dtype=mx.bfloat16)
    ids = mx.array([[256, 9], [9, 2]], dtype=mx.uint32)
    assert bool(mx.all(mx.isfinite(layer(x, ids))).item())
    assert store.layer(3) is layer


def test_grouped_prefill_routing_matches_reference(expert_checkpoint):
    path, _ = expert_checkpoint
    rng = np.random.default_rng(31)
    x = mx.array(rng.normal(size=(17, 128)), dtype=mx.float32)
    ids = mx.array(np.column_stack([rng.integers(0, 256, (17, 2)), np.full(17, 256)]), dtype=mx.uint32)
    reference = ExpertStore(path, resident=True, optimize=False).layer(3)(x, ids)
    actual = ExpertStore(path, resident=True).layer(3)(x, ids)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=5e-4, atol=2e-7)
