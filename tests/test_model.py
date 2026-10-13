import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from exlr8.model import Attention, DecoderLayer, Model, ModelArgs, load_model, route_experts


class Linear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = mx.array(weight)

    def __call__(self, x):
        return x @ self.weight.T

    def dequantize(self):
        return self.weight


class Embedding(Linear):
    def __call__(self, ids):
        return self.weight[ids]


class Experts(nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate, self.up, self.down = mx.array(gate), mx.array(up), mx.array(down)

    def __call__(self, x, ids):
        values = mx.stack([
            (nn.silu(x @ self.gate[e].T) * (x @ self.up[e].T)) @ self.down[e].T
            for e in range(self.gate.shape[0])
        ], axis=-2)
        return mx.take_along_axis(values, ids[..., None], axis=-2)


def fixtures(layers=1, topk=3):
    args = ModelArgs(
        hidden_size=8, vocab_size=17, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=2, q_lora_rank=6,
        kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4,
        v_head_dim=6, intermediate_size=12, moe_intermediate_size=8,
        n_routed_experts=4, num_experts_per_tok=2, first_k_dense_replace=1,
        index_head_dim=8, index_n_heads=2, index_topk=topk,
        indexer_types=["full"] + ["shared"] * (layers - 1),
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
    )
    rng = np.random.default_rng(23)
    def matrix(out_features, in_features):
        return Linear(rng.normal(0, 0.15, (out_features, in_features)).astype(np.float32))
    weights = {
        "self_attn.q_a_proj": matrix(6, 8),
        "self_attn.q_b_proj": matrix(16, 6),
        "self_attn.kv_a_proj_with_mqa": matrix(12, 8),
        "self_attn.kv_b_proj": matrix(20, 8),
        "self_attn.o_proj": matrix(8, 12),
        "self_attn.q_a_layernorm.weight": mx.ones(6),
        "self_attn.kv_a_layernorm.weight": mx.ones(8),
        "self_attn.indexer.wq_b": matrix(16, 6),
        "self_attn.indexer.wk": matrix(8, 8),
        "self_attn.indexer.weights_proj": matrix(2, 8),
        "self_attn.indexer.k_norm.weight": mx.array(rng.uniform(0.7, 1.3, 8).astype(np.float32)),
        "self_attn.indexer.k_norm.bias": mx.array(rng.normal(0, 0.1, 8).astype(np.float32)),
        "input_layernorm.weight": mx.ones(8),
        "post_attention_layernorm.weight": mx.ones(8),
        "mlp.gate_proj": matrix(12, 8),
        "mlp.up_proj": matrix(12, 8),
        "mlp.down_proj": matrix(8, 12),
        "mlp.gate": matrix(4, 8),
        "mlp.gate.e_score_correction_bias": mx.array([0.05, -0.04, 0.01, 0.03]),
    }
    experts = Experts(
        rng.normal(0, 0.1, (5, 8, 8)).astype(np.float32),
        rng.normal(0, 0.1, (5, 8, 8)).astype(np.float32),
        rng.normal(0, 0.1, (5, 8, 8)).astype(np.float32),
    )
    vocab = {
        "embed_tokens": Embedding(rng.normal(0, 0.4, (17, 8)).astype(np.float32)),
        "lm_head": matrix(17, 8), "norm.weight": mx.ones(8),
    }
    return args, weights, experts, vocab


def expanded_reference(args, weights, x, shared_indices=None):
    """Independent NumPy inference with full per-head K/V and causal attention."""
    def linear(name, value):
        return value @ np.asarray(weights[name].weight).T
    def rms(value, name):
        return value / np.sqrt(np.mean(value**2, axis=-1, keepdims=True) + 1e-6) * np.asarray(weights[name])
    def rope(value):
        dims = args.qk_rope_head_dim
        freq = args.rope_parameters["rope_theta"] ** (-np.arange(0, dims, 2, dtype=np.float32) / dims)
        angle = np.arange(value.shape[-2], dtype=np.float32)[:, None] * freq
        out = value.copy()
        even, odd = value[..., :dims:2], value[..., 1:dims:2]
        out[..., :dims:2] = even * np.cos(angle) - odd * np.sin(angle)
        out[..., 1:dims:2] = even * np.sin(angle) + odd * np.cos(angle)
        return out
    batch, length, _ = x.shape
    qr = rms(linear("self_attn.q_a_proj", x), "self_attn.q_a_layernorm.weight")
    q = linear("self_attn.q_b_proj", qr).reshape(batch, length, 2, 8).transpose(0, 2, 1, 3)
    q_nope, q_pe = q[..., :4], rope(q[..., 4:])
    compressed = linear("self_attn.kv_a_proj_with_mqa", x)
    latent = rms(compressed[..., :8], "self_attn.kv_a_layernorm.weight")
    k_pe = rope(compressed[..., 8:][:, None, :, :])
    expanded = linear("self_attn.kv_b_proj", latent).reshape(batch, length, 2, 10).transpose(0, 2, 1, 3)
    keys = np.concatenate([expanded[..., :4], np.broadcast_to(k_pe, (batch, 2, length, 4))], axis=-1)
    queries = np.concatenate([q_nope, q_pe], axis=-1)
    causal = np.arange(length)[:, None] >= np.arange(length)[None, :]
    selected = shared_indices
    if selected is None and length > args.index_topk:
        iq = linear("self_attn.indexer.wq_b", qr).reshape(batch, length, 2, 8).transpose(0, 2, 1, 3)
        ik = linear("self_attn.indexer.wk", x)
        ik = (ik - ik.mean(-1, keepdims=True)) / np.sqrt(ik.var(-1, keepdims=True) + 1e-6)
        ik = ik * np.asarray(weights["self_attn.indexer.k_norm.weight"]) + np.asarray(weights["self_attn.indexer.k_norm.bias"])
        iq, ik = rope(iq), rope(ik[:, None, :, :])
        index_scores = np.maximum(iq @ ik.swapaxes(-1, -2) / np.sqrt(8), 0)
        head_weights = linear("self_attn.indexer.weights_proj", x).transpose(0, 2, 1) / np.sqrt(2)
        index_scores = (index_scores * head_weights[..., None]).sum(1, keepdims=True)
        index_scores = np.where(causal, index_scores, -np.inf)
        selected = np.argpartition(-index_scores, args.index_topk - 1, axis=-1)[..., :args.index_topk]
    allowed = np.broadcast_to(causal, (batch, 1, length, length)).copy()
    if selected is not None:
        sparse = np.zeros_like(allowed)
        np.put_along_axis(sparse, selected, True, axis=-1)
        allowed &= sparse
    scores = (queries @ keys.swapaxes(-1, -2)) / np.sqrt(8)
    scores = np.where(allowed, scores, -np.inf)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    output = (probs @ expanded[..., 4:]).transpose(0, 2, 1, 3).reshape(batch, length, 12)
    return linear("self_attn.o_proj", output), selected


@pytest.mark.parametrize("length,topk", [(1, 3), (5, 8), (11, 20), (11, 3)])
def test_attention_matches_expanded_reference(length, topk):
    args, weights, _, _ = fixtures(topk=topk)
    x = np.random.default_rng(1).normal(0, 1, (2, length, 8)).astype(np.float32)
    expected, expected_ids = expanded_reference(args, weights, x)
    mask = mx.arange(length)[:, None] >= mx.arange(length)[None, :] if length > 1 else None
    actual, ids = Attention(args, 0, weights)(mx.array(x), mask)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-6)
    if expected_ids is not None:
        np.testing.assert_array_equal(np.sort(np.asarray(ids), axis=-1), np.sort(expected_ids, axis=-1))


@pytest.fixture
def joined_projection_weights():
    from exlr8.dense import DenseLinear

    args = ModelArgs(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                     q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16,
                     qk_rope_head_dim=16, v_head_dim=16, indexer_types=["full", "shared"])
    rng = np.random.default_rng(67)

    def fp8(rows):
        codes = rng.integers(8, 119, (rows, 128), dtype=np.uint8)
        codes |= rng.integers(0, 2, codes.shape, dtype=np.uint8) * 128
        return DenseLinear((3, rows, 128, 0, 0), {
            "fp8": mx.array(codes), "e8m0": mx.array(rng.integers(119, 126, (rows, 4), dtype=np.uint8))})

    def bf16(rows, columns):
        return DenseLinear((6, rows, columns, 0, 0), {
            "bf16": mx.array(rng.normal(0, 0.1, (rows, columns)), dtype=mx.bfloat16)})

    return args, {
        "self_attn.q_a_proj": fp8(32), "self_attn.kv_a_proj_with_mqa": fp8(32),
        "self_attn.q_b_proj": bf16(64, 32), "self_attn.kv_b_proj": bf16(64, 16),
        "self_attn.o_proj": bf16(128, 32),
        "self_attn.q_a_layernorm.weight": mx.ones(32, dtype=mx.bfloat16),
        "self_attn.kv_a_layernorm.weight": mx.ones(16, dtype=mx.bfloat16),
    }


def test_joined_input_projection_bits_and_parameter_bytes(joined_projection_weights):
    args, weights = joined_projection_weights
    baseline = Attention(args, 1, {**weights, "eh_proj": object()})
    before = sum(value.nbytes for _, value in tree_flatten(baseline.parameters()))
    q, kv = weights["self_attn.q_a_proj"], weights["self_attn.kv_a_proj_with_mqa"]
    original = [np.array(value) for value in (q.packed, q.exponents, kv.packed, kv.exponents)]
    attention = Attention(args, 1, weights)
    assert attention.q_a_proj is q and attention.kv_a_proj is kv
    assert callable(attention._joined_input)
    assert sum(value.nbytes for _, value in tree_flatten(attention.parameters())) == before
    assert "_joined_input" not in attention.parameters()
    mx.eval(attention.parameters())
    for value, expected in zip((q.packed, q.exponents, kv.packed, kv.exponents), original):
        np.testing.assert_array_equal(np.array(value), expected)
    x = mx.array(np.random.default_rng(68).normal(size=(1, 1, 128)), dtype=mx.bfloat16)
    x = mx.fast.rms_norm(x, mx.ones(128, dtype=mx.bfloat16), args.rms_norm_eps)
    qa, kva = attention._joined_input(x)
    for actual, expected in ((qa, q(x)), (kva, kv(x))):
        np.testing.assert_array_equal(np.array(actual.view(mx.uint16)), np.array(expected.view(mx.uint16)))


@pytest.mark.parametrize("batch,rows,dtype,use_joined", [
    (1, 1, mx.bfloat16, True), (1, 2, mx.bfloat16, False),
    (2, 1, mx.bfloat16, False), (1, 1, mx.float32, False),
])
def test_joined_input_projection_call_bounds(joined_projection_weights, monkeypatch, batch, rows, dtype, use_joined):
    args, weights = joined_projection_weights
    attention = Attention(args, 1, weights)
    joined, calls = attention._joined_input, []

    def record(x):
        calls.append(x.shape)
        return joined(x)

    monkeypatch.setattr(attention, "_joined_input", record)
    x = mx.array(np.random.default_rng(69).normal(size=(batch, rows, 128)), dtype=dtype)
    mask = mx.arange(rows)[:, None] >= mx.arange(rows)[None, :] if rows > 1 else None
    actual, _ = attention(x, mask)
    monkeypatch.setattr(attention, "_joined_input", None)
    expected, _ = attention(x, mask)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)).view(np.uint32),
                                  np.array(expected.astype(mx.float32)).view(np.uint32))
    assert calls == ([x.shape] if use_joined else [])


@pytest.mark.parametrize("case", ["no_optimize", "mtp", "nonaligned", "different_input",
                                  "other_format", "callable", "outside_main"])
def test_joined_input_projection_init_bounds(joined_projection_weights, case):
    from exlr8.dense import DenseLinear

    args, weights = joined_projection_weights
    q = weights["self_attn.q_a_proj"]
    if case == "no_optimize":
        q.optimize = False
    elif case == "mtp":
        weights["eh_proj"] = object()
    elif case == "nonaligned":
        weights["self_attn.q_a_proj"] = DenseLinear((3, 31, 128, 0, 0), {
            "fp8": q.packed.view(mx.uint8)[:31], "e8m0": q.exponents[:31]})
    elif case == "different_input":
        weights["self_attn.kv_a_proj_with_mqa"] = DenseLinear((3, 32, 160, 0, 0), {
            "fp8": mx.ones((32, 160), dtype=mx.uint8), "e8m0": mx.full((32, 5), 127, dtype=mx.uint8)})
    elif case == "other_format":
        weights["self_attn.q_a_proj"] = DenseLinear((6, 32, 128, 0, 0), {"bf16": mx.zeros((32, 128))})
    elif case == "callable":
        weights["self_attn.q_a_proj"] = Linear(np.zeros((32, 128), dtype=np.float32))
        weights["self_attn.q_a_proj"].optimize, weights["self_attn.q_a_proj"].format = True, 3
    else:
        args.num_hidden_layers = 1
    attention = Attention(args, 1, weights)
    assert attention._joined_input is None
    assert attention.q_a_proj is weights["self_attn.q_a_proj"]
    assert attention.kv_a_proj is weights["self_attn.kv_a_proj_with_mqa"]


def test_shared_layer_reuses_previous_indices():
    args, weights, _, _ = fixtures(layers=2, topk=3)
    x = np.random.default_rng(6).normal(0, 1, (2, 7, 8)).astype(np.float32)
    mask = mx.arange(7)[:, None] >= mx.arange(7)[None, :]
    _, ids = Attention(args, 0, weights)(mx.array(x), mask)
    actual, shared = Attention(args, 1, weights)(mx.array(x * 1.7), mask, prev_topk_indices=ids)
    expected, _ = expanded_reference(args, weights, x * 1.7, np.asarray(ids))
    assert shared is ids
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-6)


def test_cache_matches_prompt_and_token_processing():
    args, weights, experts, vocab = fixtures(layers=2)
    model = Model(args, [DecoderLayer(args, 0, weights), DecoderLayer(args, 1, weights, experts)], vocab)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 5, 7, 9, 11, 13, 15, 16, 2]])
    expected = model(tokens)
    cache = model.make_cache()
    outputs = [model(tokens[:, :4], cache=cache)]
    outputs += [model(tokens[:, i:i + 1], cache=cache) for i in range(4, tokens.shape[1])]
    actual = mx.concatenate(outputs, axis=1)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-5, atol=3e-6)
    assert [len(c.caches) for c in cache] == [2, 1]
    assert cache[0][0].offset == cache[0][1].offset == cache[1][0].offset == 9
    assert cache[0][0].keys.shape[-1] == args.kv_lora_rank
    assert cache[0][0].values.shape[-1] == args.qk_rope_head_dim
    assert cache[0][1].values.shape[-1] == 0
    assert tree_flatten(model.trainable_parameters()) == []
    with pytest.raises(ValueError, match="one entry"):
        model(tokens, cache=cache[:1])


def test_hidden_forward_matches_normal_logits():
    args, weights, experts, vocab = fixtures(layers=2)
    model = Model(args, [DecoderLayer(args, 0, weights), DecoderLayer(args, 1, weights, experts)], vocab)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    hidden, indices = model.forward_hidden(tokens)
    assert hidden.shape == (1, 5, args.hidden_size)
    assert indices.shape == (1, 1, 5, args.index_topk)
    np.testing.assert_array_equal(np.asarray(model.project_hidden(hidden)), np.asarray(model(tokens)))


def test_shared_sparse_decode_after_2048_cached_tokens():
    from mlx_lm.models.cache import CacheList, KVCache

    args, weights, _, _ = fixtures(layers=2, topk=2048)
    weights["self_attn.indexer.weights_proj"] = Linear(np.full((2, 8), -0.1, dtype=np.float32))
    x = np.random.default_rng(41).normal(2, 1, (1, 2051, 8)).astype(np.float32)
    full, shared = Attention(args, 0, weights), Attention(args, 1, weights)
    full_cache, shared_cache = CacheList(KVCache(), KVCache()), CacheList(KVCache())
    mask = mx.arange(2048)[:, None] >= mx.arange(2048)[None, :]
    prefill, indices = full(mx.array(x[:, :2048]), mask, full_cache)
    shared_prefill, _ = shared(mx.array(x[:, :2048] * 1.7), mask, shared_cache, indices)
    mx.eval(prefill, shared_prefill)
    assert indices is None
    for position in range(2048, 2051):
        output, indices = full(mx.array(x[:, position:position + 1]), cache=full_cache)
        shared_output, reused = shared(mx.array(x[:, position:position + 1] * 1.7), cache=shared_cache, prev_topk_indices=indices)
        assert reused is indices
        assert indices.shape == (1, 1, 1, 2048)
        expected, selected = expanded_reference(args, weights, x[:, :position + 1])
        shared_expected, _ = expanded_reference(args, weights, x[:, :position + 1] * 1.7, selected)
        np.testing.assert_allclose(np.asarray(output), expected[:, -1:], rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(np.asarray(shared_output), shared_expected[:, -1:], rtol=3e-5, atol=3e-6)
    assert full_cache[0].offset == full_cache[1].offset == shared_cache[0].offset == 2051


def test_server_prefix_cache_copies_and_trims_independent_requests():
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

    args, weights, experts, vocab = fixtures(layers=2)
    model = Model(args, [DecoderLayer(args, 0, weights), DecoderLayer(args, 1, weights, experts)], vocab)
    prefix = [1, 2, 3, 4, 5]
    original = make_prompt_cache(model)
    mx.eval(model(mx.array([prefix]), cache=original))
    saved_keys = np.asarray(original[0][0].keys).copy()
    lru = LRUPromptCache(max_size=2)
    lru.insert_cache("glm-test", prefix, original)
    for tokens, reused_length in ((prefix + [6, 7], 5), (prefix + [8, 9], 5), ([1, 2, 3, 10], 3)):
        copied, rest = lru.fetch_nearest_cache("glm-test", tokens)
        assert len(tokens) - len(rest) == reused_length
        assert [group[0].offset for group in copied] == [reused_length] * 2
        actual = model(mx.array([rest]), cache=copied)
        expected = model(mx.array([tokens]))[:, reused_length:]
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-5, atol=3e-6)
        assert [group[0].offset for group in original] == [5, 5]
        assert original[0][1].offset == 5
        np.testing.assert_array_equal(np.asarray(original[0][0].keys), saved_keys)


def test_router_bias_changes_selection_but_not_probability():
    args, _, _, _ = fixtures()
    logits = mx.array([[3.0, 2.0, -1.0, -2.0]])
    ids, weights = route_experts(logits, mx.array([0.0, 0.0, 2.0, 3.0]), args)
    chosen = np.asarray(ids)[0]
    np.testing.assert_array_equal(np.sort(chosen), [2, 3])
    probabilities = 1 / (1 + np.exp(-np.asarray(logits)[0, chosen]))
    np.testing.assert_allclose(np.asarray(weights)[0], probabilities / probabilities.sum() * 2.5, rtol=1e-6)


def test_group_router_excludes_unselected_groups():
    args, _, _, _ = fixtures()
    args.n_group, args.topk_group = 2, 1
    ids, _ = route_experts(mx.array([[4.0, 3.0, 2.0, 1.0]]), mx.zeros(4), args)
    np.testing.assert_array_equal(np.sort(np.asarray(ids)[0]), [0, 1])


def test_glm_53_indexer_schedule_and_dense_layers():
    args = ModelArgs()
    assert args.indexer_types[:7] == ["full", "full", "full", "shared", "shared", "shared", "full"]
    assert args.indexer_types.count("full") == 21
    assert args.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 75
    with pytest.raises(ValueError, match="first layer"):
        ModelArgs(num_hidden_layers=1, indexer_types=["shared"])


class RouterWeight:
    def __init__(self, dtype):
        self.dtype, self.casts = dtype, []

    def astype(self, dtype):
        self.casts.append(dtype)
        return RouterWeight(dtype)


@pytest.fixture
def router_init(monkeypatch):
    import exlr8.model as model_module

    monkeypatch.setattr(model_module, "Attention", lambda *args: object())
    return ModelArgs(), {
        "input_layernorm.weight": object(), "post_attention_layernorm.weight": object(),
        "mlp.gate.e_score_correction_bias": object(),
    }


@pytest.mark.parametrize("optimize,format,index,mtp,cast", [
    (True, 6, 3, False, True),
    (False, 6, 3, False, False),
    (True, 7, 3, False, False),
    (True, 6, 78, False, False),
    (True, 6, 78, True, False),
])
def test_main_router_caches_only_optimized_bf16_weight(router_init, optimize, format, index, mtp, cast):
    from dataclasses import replace
    from exlr8.dense import DenseLinear

    args, weights = router_init
    if index == 78:
        if mtp:
            args = replace(args, num_hidden_layers=79,
                           indexer_types=[*args.indexer_types, "full"],
                           mlp_layer_types=[*args.mlp_layer_types, "sparse"])
            weights["eh_proj"] = object()
        else:
            args.mlp_layer_types.append("sparse")
    weight = RouterWeight(mx.bfloat16 if format == 6 else mx.float32)
    router = DenseLinear((format, 256, 6144, 0, 0),
                         {"bf16" if format == 6 else "fp32": weight}, optimize=optimize)
    other_weight = RouterWeight(mx.bfloat16)
    weights["self_attn.indexer.weights_proj"] = DenseLinear((6, 32, 6144, 0, 0), {"bf16": other_weight})
    weights["mlp.gate"] = router
    layer = DecoderLayer(args, index, weights, object())
    assert layer.gate is router
    assert weight.casts == ([mx.float32] if cast else [])
    assert layer.gate.weight.dtype == (mx.float32 if cast else weight.dtype)
    assert (layer.gate.weight is weight) == (not cast)
    assert other_weight.dtype == mx.bfloat16 and other_weight.casts == []


@pytest.mark.parametrize("tagged", [False, True])
def test_router_init_keeps_simple_callables(router_init, tagged):
    args, weights = router_init
    router = lambda x: x
    if tagged:
        router.optimize, router.format, router.weight = True, 6, RouterWeight(mx.bfloat16)
    weights["mlp.gate"] = router
    layer = DecoderLayer(args, 3, weights, object())
    assert layer.gate is router
    if tagged:
        assert router.weight.dtype == mx.bfloat16 and router.weight.casts == []


@pytest.mark.parametrize("batch,rows,k,resident,optimize,routed,shared,use_finish", [
    (1, 1, 4, True, True, 8, 1, True),
    (2, 1, 4, True, True, 8, 1, False),
    (1, 2, 4, True, True, 8, 1, False),
    (1, 1, 3, True, True, 8, 1, False),
    (1, 1, 4, False, True, 8, 1, False),
    (1, 1, 4, True, False, 8, 1, False),
    (1, 1, 4, True, True, 2, 1, False),
    (1, 1, 4, True, True, 8, 2, False),
])
def test_decoder_finish_dispatch(monkeypatch, batch, rows, k, resident, optimize, routed, shared, use_finish):
    from types import SimpleNamespace
    import exlr8.fast_decode as decode
    import exlr8.model as model_module

    args, weights, experts, _ = fixtures(layers=2)
    args.num_experts_per_tok, args.n_shared_experts = routed, shared
    experts.k = k
    experts.store = SimpleNamespace(resident=resident, optimize=optimize)
    layer = DecoderLayer(args, 1, weights, experts)
    compiled, calls, native_calls = [], [], []
    original = Experts.__call__

    def native(expert, x, ids):
        native_calls.append(ids.shape)
        return original(expert, x, ids)

    def build(expert):
        compiled.append(expert)
        def finish(x, ids, scores, residual):
            calls.append((x, ids, scores, residual))
            return mx.full(residual.shape, 0.75, dtype=residual.dtype)
        return finish

    monkeypatch.setattr(Experts, "__call__", native)
    monkeypatch.setattr(decode, "compiled_expert_mlp_finish", build)
    monkeypatch.setattr(model_module, "route_experts", lambda logits, bias, a: (
        mx.zeros(logits.shape[:-1] + (routed,), dtype=mx.uint32),
        mx.ones(logits.shape[:-1] + (routed,), dtype=mx.float32)))
    mask, cache, previous, topk = object(), object(), object(), object()
    def attention(x, passed_mask, passed_cache, passed_previous):
        assert (passed_mask, passed_cache, passed_previous) == (mask, cache, previous)
        return mx.zeros_like(x), topk
    layer.self_attn = attention
    x = mx.ones((batch, rows, args.hidden_size))
    for _ in range(2):
        out, indices = layer(x, mask, cache, previous)
        assert indices is topk
        assert out.shape == x.shape
    assert len(compiled) == int(use_finish)
    assert len(calls) == (2 if use_finish else 0)
    assert len(native_calls) == (0 if use_finish else 2)
    if use_finish:
        normalized, ids, scores, residual = calls[0]
        assert ids.shape == (1, 1, 9) and scores.shape == (1, 1, 8)
        np.testing.assert_array_equal(np.array(residual), np.array(x))
        np.testing.assert_array_equal(np.array(normalized), np.array(mx.fast.rms_norm(x, layer.post_norm, args.rms_norm_eps)))
        np.testing.assert_array_equal(np.array(out), np.full(x.shape, 0.75, dtype=np.float32))


@pytest.mark.parametrize("enabled,batch,rows,width,dtype,residual_dtype,expected", [
    (True, 1, 1, 6144, "bfloat16", "bfloat16", True),
    (False, 1, 1, 6144, "bfloat16", "bfloat16", False),
    (True, 2, 1, 6144, "bfloat16", "bfloat16", False),
    (True, 1, 2, 6144, "bfloat16", "bfloat16", False),
    (True, 1, 1, 128, "bfloat16", "bfloat16", False),
    (True, 1, 1, 6144, "float32", "float32", False),
    (True, 1, 1, 6144, "bfloat16", "float32", False),
])
def test_decoder_residual_rms_dispatch_bounds(monkeypatch, enabled, batch, rows, width, dtype, residual_dtype, expected):
    import exlr8.fast_norm as norm

    args, weights, _, _ = fixtures()
    layer = DecoderLayer(args, 0, weights)
    layer.input_norm = layer.post_norm = mx.ones(width, dtype=getattr(mx, dtype))
    layer._fused_rms = enabled
    layer.self_attn = lambda x, *args: (mx.zeros(x.shape, dtype=getattr(mx, residual_dtype)), None)
    layer.gate_proj = layer.up_proj = lambda x: x
    layer.down_proj = lambda x: mx.zeros_like(x)
    calls = []
    def pair(x, residual, weight):
        calls.append(x.shape)
        h = x + residual
        return h, mx.fast.rms_norm(h, weight, 1e-5)
    monkeypatch.setattr(norm, "residual_rms_pair", pair)
    x = mx.ones((batch, rows, width), dtype=getattr(mx, dtype))
    out, _ = layer(x)
    np.testing.assert_array_equal(np.array(out.astype(mx.float32)), np.array(x.astype(mx.float32)))
    assert len(calls) == int(expected)


@pytest.mark.parametrize("optimize,indexed_prefill", [(True, True), (True, False), (False, True)])
def test_loader_reads_main_layers_and_reports_progress(tmp_path, monkeypatch, optimize, indexed_prefill):
    import exlr8.dense
    import exlr8.dsa_indexed

    args, weights, experts, vocab = fixtures(layers=2)
    (tmp_path / "config.json").write_text(json.dumps(vars(args)))
    loaded, indexed = [], []
    class Store:
        def __init__(self, path, optimize=True, decode_k4=False):
            assert path == tmp_path
            assert not decode_k4
        def layer(self, index):
            loaded.append(index)
            return weights
        def vocab(self):
            return vocab
    class ExpertStore:
        def layer(self, index):
            assert index == 1
            return experts
    monkeypatch.setattr(exlr8.dense, "DenseStore", Store)
    monkeypatch.setattr(exlr8.dsa_indexed, "install_indexed_prefill", indexed.append)
    progress = []
    model = load_model(tmp_path, ExpertStore(), lambda current, total: progress.append((current, total)),
                       optimize=optimize, indexed_prefill=indexed_prefill)
    assert loaded == [0, 1]
    assert progress == [(1, 2), (2, 2)]
    assert len(model.layers) == 2
    assert indexed == ([layer.self_attn for layer in model.layers] if optimize and indexed_prefill else [])
    assert model(mx.array([[1]])).shape == (1, 1, 17)
