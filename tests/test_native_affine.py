import os
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.activations import swiglu
import numpy as np
import pytest

from exlr8.fast_decode import compiled_expert_mlp_finish
from exlr8.native_affine import NativeAffine
from exlr8.trellis import decode_trellis_reference, hadamard_reference


pytestmark = pytest.mark.skipif(
    os.environ.get("EXLR8_TEST_METAL") != "1", reason="set EXLR8_TEST_METAL=1 on a Metal host",
)


@pytest.fixture(scope="module")
def native_expert():
    rng = np.random.default_rng(819)
    h, m, count = 256, 128, 257
    raw, scales = {}, {}
    size = h * m // 2
    for projection in ("gate", "up", "down"):
        incoming, outgoing = (m, h) if projection == "down" else (h, m)
        raw[projection] = rng.integers(0, 256, (count, size), np.uint8)
        scales[projection] = (
            mx.array(rng.uniform(-.04, .04, (count, incoming)), mx.float16),
            mx.array(rng.uniform(-.5, .5, (count, outgoing)), mx.float16),
        )
    expert = SimpleNamespace(hidden=h, intermediate=m, expert_bytes=size, scales=scales,
                             packed={p: mx.array(a) for p, a in raw.items()})
    expert.affine = NativeAffine(expert)
    matrices = {
        p: {i: np.asarray(mx.dequantize(
            *(a[i:i + 1] for a in expert.affine.weights[p]),
            group_size=128, bits=4, mode="affine", dtype=mx.float32,
        )[0]) for i in (0, 1, 255, 256)} for p in raw
    }
    return expert, raw, matrices


def test_reencoding_keeps_matrix_orientation_expert_offsets_and_e4m3(native_expert):
    expert, raw, _ = native_expert
    for projection in ("gate", "up", "down"):
        h, m = expert.hidden, expert.intermediate
        incoming, outgoing, side = (m, h, 2) if projection == "down" else (h, m, 1)
        for i in (0, 1, 255, 256):
            decoded = decode_trellis_reference(raw[projection][i], 4, incoming, outgoing, side).T
            expected = mx.quantize(mx.array(decoded, mx.float32)[None], group_size=128, bits=4, mode="affine")
            for actual, wanted in zip(expert.affine.weights[projection], expected):
                np.testing.assert_array_equal(np.asarray(actual[i:i + 1]), np.asarray(wanted))


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 17])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_native_slots_match_dequantized_matrix_reference(native_expert, rows, dtype):
    expert, _, matrices = native_expert
    rng = np.random.default_rng(820 + rows)
    x = mx.array(rng.normal(size=(1, rows, expert.hidden)), dtype)
    ids = mx.array(rng.choice([0, 1, 255, 256], (1, rows, 9)), mx.uint32)
    scale = {p: tuple(np.asarray(a.astype(mx.float32)) for a in pair) for p, pair in expert.scales.items()}
    expected = []
    for value, row_ids in zip(np.asarray(x.astype(mx.float32))[0], ids.tolist()[0]):
        outputs = []
        for i in row_ids:
            gu = []
            for p in ("gate", "up"):
                inner = hadamard_reference(value * scale[p][0][i]) @ matrices[p][i].T
                gu.append(mx.array(hadamard_reference(inner) * scale[p][1][i], dtype))
            activation = np.asarray(swiglu(*gu).astype(mx.float32))
            inner = hadamard_reference(activation * scale["down"][0][i]) @ matrices["down"][i].T
            outputs.append(mx.array(hadamard_reference(inner) * scale["down"][1][i], dtype))
        expected.append(mx.stack(outputs))
    actual = expert.affine.slots(x, ids)
    wanted = mx.stack(expected)[None]
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), np.asarray(wanted.astype(mx.float32)),
                               rtol=.01 if dtype == mx.bfloat16 else 3e-4, atol=2e-6)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_compiled_native_finish_keeps_routed_shared_and_residual_casts(native_expert, rows):
    expert, _, _ = native_expert
    rng = np.random.default_rng(831 + rows)
    x = mx.array(rng.normal(size=(1, rows, expert.hidden)), mx.bfloat16)
    ids = mx.array(rng.choice([0, 1, 255], (1, rows, 9)), mx.uint32)
    ids[..., -1] = 256
    scores = mx.array(rng.uniform(.03, .2, (1, rows, 8)), mx.float32)
    residual = mx.array(rng.normal(size=x.shape), mx.bfloat16)
    values = expert.affine.slots(x, ids)
    routed = (values[..., :-1, :] * scores[..., None]).sum(axis=-2).astype(x.dtype)
    wanted = residual + (routed + values[..., -1, :])
    finish = compiled_expert_mlp_finish(expert)
    assert finish is expert.affine.finish
    actual = finish(x, ids, scores, residual)
    assert actual.shape == residual.shape and actual.dtype == residual.dtype
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), np.asarray(wanted.astype(mx.float32)))
