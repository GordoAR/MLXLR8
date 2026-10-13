"""The public drafter must receive its trained GLM layer inputs."""

import copy

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mlx_dspark")

from mlx_dspark.config import DSparkConfig
from mlx_dspark.model import DSparkDrafter

from exlr8.model import DecoderLayer, Model
from test_model import fixtures
from tools.benchmark_dspark import prepare_draft_prefix, wrap_target


def tiny_target():
    args, weights, experts, vocab = fixtures(layers=2)
    return Model(args, [DecoderLayer(args, 0, weights),
                        DecoderLayer(args, 1, weights, experts)], vocab)


def test_taps_use_layer_inputs_and_keep_native_logits():
    with mx.stream(mx.cpu):
        model = tiny_target()
        captured = []
        hook = lambda index, hidden: captured.append(hidden)
        model.trace = hook
        ids = mx.array([[1, 2, 3, 4]])
        expected = model(ids, cache=model.make_cache())
        target = wrap_target(model, None)
        logits, fused = target.run(ids, target.make_cache(), [1])
        mx.eval(expected, logits, fused)
        np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(captured[0]))
        assert model.trace is hook


def test_saved_prefix_window_matches_full_context():
    with mx.stream(mx.cpu):
        model = tiny_target()
        target = wrap_target(model, None)
        config = DSparkConfig(
            family="qwen3", hidden_size=8, vocab_size=17, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=4,
            intermediate_size=12, rope_type="default", rope_theta=10000,
            attention_k_eq_v=False, block_size=4, mask_token_id=0,
            causal_block=True, sliding_window=4, target_layer_ids=[1, 2],
            has_own_embed=False, has_own_lm_head=False, markov_rank=0,
            enable_confidence_head=False)
        drafter = DSparkDrafter(config)
        cache, full_context = target.make_cache(), drafter.make_ctx_cache()
        ids = list(range(1, 13))
        for position in range(0, len(ids) - 1, 4):
            _, fused = target.prefill(mx.array([ids[position:min(position + 4, len(ids) - 1)]]),
                                      cache, config.target_layer_ids, want_logits=False)
            drafter.update_context(fused, position, full_context)
            mx.eval([group.state for group in cache], [c.k for c in full_context])
        original = copy.deepcopy(cache)
        prepared, restored, result = prepare_draft_prefix(target, drafter, cache, ids, 4)
        assert result["target_cache_bit_equal"]
        assert result["replayed_tokens"] == 7
        assert all(c.length == 11 for c in restored)
        for old, new in zip(original, prepared):
            for a, b in zip(old.caches, new.caches):
                np.testing.assert_array_equal(np.asarray(a.keys), np.asarray(b.keys))
        noise = mx.ones((1, 4, 8))
        expected = drafter.backbone(noise, 11, full_context)
        actual = drafter.backbone(noise, 11, restored)
        mx.eval(expected, actual)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
