import os
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.activations import swiglu
import numpy as np
import pytest

from exlr8.joined_mxfp4 import JoinedMXFP4
from exlr8.trellis import decode_trellis_reference, hadamard_reference


pytestmark = pytest.mark.skipif(
    os.environ.get("EXLR8_TEST_METAL") != "1", reason="set EXLR8_TEST_METAL=1 on a Metal host",
)


@pytest.fixture(scope="module")
def joined_expert():
    rng = np.random.default_rng(817)
    h, m, count = 256, 128, 257
    signs = rng.choice([-1, 1], h).astype(np.float16)
    raw, scales = {}, {}
    for projection in ("gate", "up", "down"):
        incoming, outgoing = (m, h) if projection == "down" else (h, m)
        raw[projection] = rng.integers(0, 256, (count, h * m // 2), np.uint8)
        incoming_scale = rng.uniform(.01, .04, (count, incoming)).astype(np.float16)
        if projection != "down":
            block = rng.uniform(.01, .04, (count, h // 128)).astype(np.float16)
            block *= rng.choice([-1, 1], block.shape)
            incoming_scale = np.repeat(block, 128, axis=-1) * signs
        scales[projection] = (mx.array(incoming_scale),
                              mx.array(rng.uniform(-.5, .5, (count, outgoing)).astype(np.float16)))
    expert = SimpleNamespace(hidden=h, intermediate=m, expert_bytes=h * m // 2,
                             scales=scales, packed={p: mx.array(a) for p, a in raw.items()})
    joined = JoinedMXFP4(expert)
    matrices = {p: {i: np.asarray(mx.dequantize(
        *(a[i:i + 1] if a is not None else None for a in joined.weights[p]),
        mode="mxfp4", group_size=32, bits=4, dtype=mx.float32,
    )[0]) for i in (0, 1, 255, 256)} for p in joined.weights}
    return expert, joined, raw, matrices


def test_folded_quantization_matches_cpu_trellis_orientation(joined_expert):
    expert, joined, raw, _ = joined_expert
    unit = np.asarray(joined.unit.astype(mx.float32))[0]
    for projection, offset in (("gate", 0), ("up", expert.intermediate), ("down", 0)):
        incoming, outgoing, side = ((expert.intermediate, expert.hidden, 2)
                                    if projection == "down" else (expert.hidden, expert.intermediate, 1))
        for i in (0, 1, 255, 256):
            matrix = decode_trellis_reference(raw[projection][i], 4, incoming, outgoing, side)
            if projection != "down":
                factor = np.asarray(expert.scales[projection][0].astype(mx.float32))[i] * unit
                matrix *= factor[:, None]
            expected = mx.quantize(mx.array(matrix.T[None], mx.float32), mode="mxfp4")
            actual = joined.weights["down" if projection == "down" else "gate_up"]
            for left, right in zip(actual[:2], expected):
                np.testing.assert_array_equal(np.asarray(left[i:i + 1, offset:offset + outgoing]), np.asarray(right))


@pytest.mark.parametrize("rows", [1, 4, 115])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_joined_slots_match_independent_transforms_and_dequantized_weights(joined_expert, rows, dtype):
    expert, joined, _, matrices = joined_expert
    rng = np.random.default_rng(840 + rows)
    x = mx.array(rng.normal(size=(1, rows, expert.hidden)), dtype)
    ids = mx.array(rng.choice([0, 1, 255, 256], (1, rows, 9)), mx.uint32)
    unit = np.asarray(joined.unit.astype(mx.float32))[0]
    scales = {p: tuple(np.asarray(a.astype(mx.float32)) for a in pair) for p, pair in expert.scales.items()}
    expected = []
    for value, row_ids in zip(np.asarray(x.astype(mx.float32))[0], ids.tolist()[0]):
        outputs = []
        common = hadamard_reference(value * unit)
        for i in row_ids:
            gu = common @ matrices["gate_up"][i].T
            gate = mx.array(hadamard_reference(gu[:expert.intermediate]) * scales["gate"][1][i], dtype)
            up = mx.array(hadamard_reference(gu[expert.intermediate:]) * scales["up"][1][i], dtype)
            activation = np.asarray(swiglu(gate, up).astype(mx.float32))
            inner = hadamard_reference(activation * scales["down"][0][i]) @ matrices["down"][i].T
            outputs.append(mx.array(hadamard_reference(inner) * scales["down"][1][i], dtype))
        expected.append(mx.stack(outputs))
    actual, wanted = joined.slots(x, ids), mx.stack(expected)[None]
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), np.asarray(wanted.astype(mx.float32)),
                               rtol=.01 if dtype == mx.bfloat16 else 3e-4, atol=2e-6)


@pytest.mark.parametrize("rows", [1, 4])
def test_joined_finish_keeps_shared_routed_and_residual_casts(joined_expert, rows):
    expert, joined, _, _ = joined_expert
    rng = np.random.default_rng(861)
    x = mx.array(rng.normal(size=(1, rows, expert.hidden)), mx.bfloat16)
    ids = mx.array(rng.choice([0, 1, 255], (1, rows, 9)), mx.uint32)
    ids[..., -1] = 256
    scores = mx.array(rng.uniform(.03, .2, (1, rows, 8)), mx.float32)
    residual = mx.array(rng.normal(size=x.shape), mx.bfloat16)
    values = joined.slots(x, ids)
    routed = (values[..., :-1, :] * scores[..., None]).sum(axis=-2).astype(x.dtype)
    wanted = residual + (routed + values[..., -1, :])
    actual = joined.finish(x, ids, scores, residual)
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint16)), np.asarray(wanted.view(mx.uint16)))
