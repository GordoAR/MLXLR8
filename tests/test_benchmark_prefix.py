"""A saved benchmark prefix must restore its token and MTP cache state."""

import mlx.core as mx
from mlx_lm.models.cache import CacheList, KVCache, save_prompt_cache, trim_prompt_cache
import numpy as np
import pytest

from exlr8.mtp import MTPState
from tools.benchmark_runtime import compare_verification, load_prefix, save_prefix


@pytest.mark.parametrize("mtp", [False, True])
def test_prefix_roundtrip_and_cache_extension(tmp_path, mtp):
    with mx.stream(mx.cpu):
        ids = [4, 7, 2]
        cache = [CacheList(KVCache(), KVCache())]
        if mtp:
            cache.append(MTPState())
        for group in cache:
            for item in group.caches:
                values = mx.array([[[[4.0, 0.0], [7.0, 1.0]]]])
                item.update_and_fetch(values, values)
        path = tmp_path / "prefix.safetensors"
        save_prefix(path, cache, ids, tmp_path, True)
        restored, tokens = load_prefix(path, tmp_path, True, mtp)
        assert tokens == ids
        assert len(restored) == len(cache)
        assert isinstance(restored[-1], MTPState) == mtp
        for original, loaded in zip(cache, restored):
            for old, new in zip(original.caches, loaded.caches):
                assert new.offset == 2
                np.testing.assert_array_equal(np.asarray(new.keys), np.asarray(old.keys))
                next_value = mx.array([[[[2.0, 2.0]]]])
                new.update_and_fetch(next_value, next_value)
                mx.eval(new.state)
                assert new.offset == 3
                assert old.offset == 2
        trim_prompt_cache(restored, 1)
        assert all(group[0].offset == 2 for group in restored)


def test_prefix_rejects_wrong_model_mode_and_missing_mtp(tmp_path):
    with mx.stream(mx.cpu):
        cache = [CacheList(KVCache())]
        values = mx.ones((1, 1, 2, 2))
        cache[0][0].update_and_fetch(values, values)
        path = tmp_path / "prefix.safetensors"
        save_prefix(path, cache, [1, 2, 3], tmp_path, True)
        for model_path, optimized, mtp in [(tmp_path / "other", True, False),
                                           (tmp_path, False, False), (tmp_path, True, True)]:
            with pytest.raises(ValueError):
                load_prefix(path, model_path, optimized, mtp)
        cache[0][0].trim(1)
        save_prefix(path, cache, [1, 2, 3], tmp_path, True)
        with pytest.raises(ValueError, match="offset"):
            load_prefix(path, tmp_path, True, False)


@pytest.mark.parametrize("optimized", [False, True])
def test_legacy_prefix_reuse_requires_unchanged_expert_values(tmp_path, optimized):
    with mx.stream(mx.cpu):
        cache = [CacheList(KVCache())]
        values = mx.ones((1, 1, 2, 2))
        cache[0][0].update_and_fetch(values, values)
        path = tmp_path / "legacy.safetensors"
        save_prompt_cache(str(path), cache, metadata={
            "model": str(tmp_path.resolve()), "prompt_ids": "[1, 2, 3]",
            "optimized": str(optimized), "mtp": "False",
        })
        if optimized:
            with pytest.raises(ValueError, match="expert format"):
                load_prefix(path, tmp_path, optimized, False)
        else:
            restored, ids = load_prefix(path, tmp_path, optimized, False)
            assert ids == [1, 2, 3] and restored[0][0].offset == 2


def test_affine3_prefix_is_rejected_for_affine4(tmp_path):
    with mx.stream(mx.cpu):
        cache = [CacheList(KVCache())]
        values = mx.ones((1, 1, 2, 2))
        cache[0][0].update_and_fetch(values, values)
        path = tmp_path / "affine3.safetensors"
        save_prompt_cache(str(path), cache, metadata={
            "model": str(tmp_path.resolve()), "prompt_ids": "[1, 2, 3]",
            "optimized": "True", "mtp": "False", "expert_format": "native_affine3_group128_f32",
        })
        with pytest.raises(ValueError, match="expert format"):
            load_prefix(path, tmp_path, True, False)


def test_layer_fidelity_keeps_causal_inputs_and_restores_hooks():
    from exlr8 import model as model_module
    from test_model import fixtures

    with mx.stream(mx.cpu):
        args, weights, experts, vocab = fixtures(layers=2)
        model = model_module.Model(args, [model_module.DecoderLayer(args, 0, weights),
                                        model_module.DecoderLayer(args, 1, weights, experts)], vocab)
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), cache=cache))
        original_layer, original_router = model_module.DecoderLayer.__call__, model_module.route_experts
        result = compare_verification(model, cache, [4, 5, 6, 7, 8, 9, 10], 3, 1, layer_trace=True)
        assert result["first_token_unchanged_when_future_inputs_change"]
        assert result["cache_offsets_match"]
        assert result["relative_l2"] < 2e-6
        assert all(row["first_token_relative_l2"] < 2e-6 for row in result["layers"])
        assert all(row["first_token_dsa_set_equal"] for row in result["layers"])
        assert result["layers"][1]["first_token_expert_set_equal"]
        assert model_module.DecoderLayer.__call__ is original_layer
        assert model_module.route_experts is original_router
