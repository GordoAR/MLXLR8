import os

import mlx.core as mx
import numpy as np
import pytest

from exlr8.dsa_prefill import heads_as_rows_attention
from exlr8.dsa_prefill_splitk import native_splitk_pv, splitk_attention, splitk_pv


metal = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1", reason="set EXLR8_TEST_METAL=1 on a Metal host")


@metal
@pytest.mark.parametrize("batch,m,k,n", [(1, 13, 128, 70), (4, 64, 2048, 512)])
def test_splitk_pv_matches_float64_and_is_row_invariant(batch, m, k, n):
    rng = np.random.default_rng(103)
    p = mx.softmax(mx.array(rng.normal(size=(batch, 1, m, k)).astype(np.float32), mx.bfloat16), axis=-1, precise=True)
    v = mx.array(rng.normal(0, 0.7, size=(batch, 1, k, n)).astype(np.float32), mx.bfloat16)
    expected = np.asarray(p.astype(mx.float32)).astype(np.float64) @ np.asarray(v.astype(mx.float32)).astype(np.float64)
    expected = np.asarray(mx.array(expected.astype(np.float32), mx.bfloat16).astype(mx.float32))
    for call in (splitk_pv, native_splitk_pv):
        out = call(p, v)
        one = call(p[:1], v[:1])
        mx.eval(out, one)
        np.testing.assert_array_equal(np.asarray(out[:1].astype(mx.float32)), np.asarray(one.astype(mx.float32)))
        actual = np.asarray(out.astype(mx.float32))
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-4


@metal
def test_splitk_keeps_tiny_bf16_probabilities():
    p = mx.full((1, 1, 8, 64), 2.0**-30, mx.bfloat16)
    v = mx.ones((1, 1, 64, 16), mx.bfloat16)
    np.testing.assert_array_equal(np.asarray(splitk_pv(p, v).astype(mx.float32)), 2.0**-24)


@metal
def test_splitk_attention_keeps_selected_keys_and_causal_mask():
    rng = np.random.default_rng(104)
    rand = lambda shape: mx.array(rng.normal(0, 0.7, size=shape).astype(np.float32), mx.bfloat16)
    q, qp = rand((1, 64, 4, 512)), rand((1, 64, 4, 64))
    latent, pe = rand((1, 1, 132, 512)), rand((1, 1, 132, 64))
    indices = mx.array(np.stack([rng.choice(132, 64, replace=False) for _ in range(4)])[None, None])
    mask = mx.arange(132)[None, None, None, :] % 5 != 0
    actual = splitk_attention(q, qp, latent, pe, indices, mask, 0.0625)
    expected = heads_as_rows_attention(q, qp, latent, pe, indices, mask, 0.0625)
    a, b = [np.asarray(x.astype(mx.float32)) for x in (actual, expected)]
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 2e-4


@pytest.mark.parametrize("length,selected,non_bf16,expected", [
    (1, 64, None, "native"), (5, 64, None, "native"),
    (2, 63, None, "native"), (4, 127, None, "native"),
    (2, 64, 0, "native"), (2, 64, 1, "native"),
    (2, 64, 2, "native"), (2, 64, 3, "native"),
    (2, 64, None, "splitk"), (4, 2048, None, "splitk"),
])
def test_private_control_uses_original_for_unsupported_inputs(monkeypatch, length, selected, non_bf16, expected):
    from types import SimpleNamespace
    from exlr8 import dsa_prefill_splitk as experiment

    operands = [SimpleNamespace(shape=(1, 64, length, 512), dtype=mx.bfloat16) for _ in range(4)]
    if non_bf16 is not None:
        operands[non_bf16].dtype = mx.float32
    indices = SimpleNamespace(shape=(1, 1, length, selected))
    attention = SimpleNamespace(_absorbed_queries=lambda q: q,
                                _sparse_attention=lambda *args: "native", scale=0.0625)
    monkeypatch.setattr(experiment, "splitk_attention", lambda *args: "splitk")
    experiment.install_splitk_control(attention)
    assert attention._sparse_attention(*operands, indices, None) == expected
