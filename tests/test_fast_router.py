"""Check fused routing against independent selection and weighting."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from exlr8 import fast_router
from exlr8.dense import DenseLinear
from exlr8.model import DecoderLayer, ModelArgs


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("case", ["random", "bias", "ties", "saturated", "zero", "strided"])
def test_selection_and_weights(rows, case):
    rng = np.random.default_rng(28)
    host = rng.normal(size=(1, rows, 256)).astype(np.float32)
    bias = rng.normal(0, .2, 256).astype(np.float32)
    if case == "bias":
        bias[:8] = 4
    elif case == "ties":
        host[:] = .5
        bias[:] = 0
    elif case == "saturated":
        host *= 100
    elif case == "zero":
        host[:] = -1000
    values = mx.array(host)
    correction = mx.array(bias)
    if case == "strided":
        values = mx.stack([values, mx.zeros_like(values)], axis=-1)[..., 0]
        correction = mx.stack([correction, mx.zeros_like(correction)], axis=-1)[..., 0]
    ids, scores = fast_router.route(values, correction)
    mx.eval(ids, scores)
    chosen = np.array(ids)[..., :8]
    assert ids.shape == (1, rows, 9) and scores.shape == (1, rows, 8)
    np.testing.assert_array_equal(np.array(ids)[..., 8], 256)
    with np.errstate(over="ignore"):
        small = 1 / (1 + np.exp(np.abs(host)))
    original = np.where(host < 0, small, 1 - small)
    expected = np.argsort(-(original + bias), axis=-1, kind="stable")[..., :8]
    np.testing.assert_array_equal(chosen, expected)
    weights = np.take_along_axis(original, chosen, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * 2.5
    np.testing.assert_allclose(np.array(scores), weights, rtol=1e-6, atol=1e-7)
    assert np.isfinite(np.array(scores)).all()


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_compiled_finish_supplies_shared_expert_and_residual(monkeypatch, rows):
    from exlr8 import fast_decode

    bias = mx.linspace(-.1, .1, 256)
    gate = lambda x: x[..., :256]
    # Distinct expert values expose missing slots and incorrect score weighting.
    def finish(x, ids, scores, residual):
        result = (ids[..., :8].astype(mx.float32) * scores).sum(axis=-1)
        result += ids[..., 8].astype(mx.float32)
        return residual + result[..., None]
    monkeypatch.setattr(fast_decode, "compiled_expert_mlp_finish", lambda expert: finish)
    x = mx.linspace(-2, 2, rows * 256).reshape(1, rows, 256).astype(mx.bfloat16)
    residual = mx.full(x.shape, .25, mx.float32)
    actual = fast_router.compiled_finish(object(), gate, bias)(x, residual)
    ids, scores = fast_router.route(x.astype(mx.float32), bias)
    expected = finish(x, ids, scores, residual)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-6)


@pytest.mark.parametrize("case", ["one", "two", "three", "four", "five", "batch",
    "float", "width", "groups", "top_group", "unnormalized", "scale", "routed",
    "shared", "bias_type", "gate_type", "gate_optimize", "k3", "nonresident", "unoptimized"])
def test_decoder_dispatch_and_cached_factory(monkeypatch, case):
    from exlr8 import fast_decode
    from exlr8 import model as model_module

    rows = {"two": 2, "three": 3, "four": 4, "five": 5}.get(case, 1)
    batch, width = (2 if case == "batch" else 1), (128 if case == "width" else 6144)
    dtype = mx.float32 if case == "float" else mx.bfloat16
    args = ModelArgs()
    if case == "groups":
        args.n_group = 8
    elif case == "top_group":
        args.topk_group = 2
    elif case == "unnormalized":
        args.norm_topk_prob = False
    elif case == "scale":
        args.routed_scaling_factor = 1
    elif case == "routed":
        args.num_experts_per_tok = 4
    elif case == "shared":
        args.n_shared_experts = 2

    class Experts:
        k = 3 if case == "k3" else 4
        store = SimpleNamespace(resident=case != "nonresident", optimize=case != "unoptimized")
        def __call__(self, x, ids):
            return mx.zeros(ids.shape + (x.shape[-1],), x.dtype)

    # Build only the fields used after attention; no model weights are loaded.
    layer = DecoderLayer.__new__(DecoderLayer)
    super(DecoderLayer, layer).__init__()
    layer.args, layer.experts = args, Experts()
    layer.input_norm = layer.post_norm = mx.ones(width, dtype=dtype)
    layer._fused_rms = False
    layer._expert_finish = layer._routed_finish = None
    layer.self_attn = lambda x, *unused: (mx.zeros_like(x), "shared-indices")
    layer.gate = DenseLinear((6, 256, width, 0, 0), {"bf16": mx.zeros((256, width))})
    if case == "gate_type":
        layer.gate.weight = layer.gate.weight.astype(mx.bfloat16)
    if case == "gate_optimize":
        layer.gate.optimize = False
    layer.bias = mx.zeros(256, mx.bfloat16 if case == "bias_type" else mx.float32)
    built, calls = [], []
    def factory(expert, gate, bias):
        built.append(expert)
        def call(x, residual):
            calls.append(x.shape)
            return residual + 1
        return call
    monkeypatch.setattr(fast_router, "compiled_finish", factory)
    monkeypatch.setattr(fast_decode, "compiled_expert_mlp_finish", lambda expert:
        (lambda x, ids, scores, residual: residual))
    monkeypatch.setattr(model_module, "route_experts", lambda logits, bias, a: (
        mx.zeros(logits.shape[:-1] + (a.num_experts_per_tok,), mx.uint32),
        mx.ones(logits.shape[:-1] + (a.num_experts_per_tok,))))
    x = mx.ones((batch, rows, width), dtype)
    for _ in range(2):
        actual, indices = layer(x)
        assert indices == "shared-indices"
        expected = 2 if case in ("one", "two", "three", "four") else 1
        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), expected)
    selected = case in ("one", "two", "three", "four")
    assert len(built) == int(selected)
    assert len(calls) == 2 * int(selected)
