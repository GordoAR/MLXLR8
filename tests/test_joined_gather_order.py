import os

import mlx.core as mx
import numpy as np
import pytest

from exlr8 import joined_mxfp4 as joined


pytestmark = pytest.mark.skipif(
    os.environ.get("EXLR8_TEST_METAL") != "1", reason="set EXLR8_TEST_METAL=1 on a Metal host",
)


@pytest.mark.parametrize("k,n", [(512, 512), (1024, 768)])
def test_ordered_gather_matches_independent_weights_with_partial_tiles(k, n):
    mx.random.seed(933)
    x = mx.random.normal((4, 1, k))
    packed = tuple(mx.quantize(mx.random.normal((5, n, k)), mode="mxfp4")) + (None,)
    ids = mx.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 4], mx.uint32)
    lhs = mx.array([3, 1, 0, 2, 3, 0, 3, 2, 1, 0], mx.uint32)
    weights = mx.dequantize(*packed, mode="mxfp4")
    expected = mx.stack([x[l] @ weights[e].T for l, e in zip(lhs.tolist(), ids.tolist())])
    actual = joined._gather_ordered(x, packed, lhs, ids)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("h,m", [(512, 512), (256, 128)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_inner_dispatch_preserves_other_shapes_and_native_result(monkeypatch, rows, h, m, dtype):
    mx.random.seed(934)
    count = 17
    weights = {
        "gate_up": tuple(mx.quantize(mx.random.normal((count, 2*m, h)), mode="mxfp4")) + (None,),
        "down": tuple(mx.quantize(mx.random.normal((count, h, m)), mode="mxfp4")) + (None,),
    }
    scales = {name: (mx.full((count, incoming), .1), mx.full((count, outgoing), .1))
              for name, incoming, outgoing in (("gate", h, m), ("up", h, m), ("down", m, h))}
    x = mx.random.normal((1, rows, h)).astype(dtype)
    ids = mx.array([[(r+i) % count for i in range(9)] for r in range(rows)], mx.uint32)[None]
    unit = mx.ones((1, h))
    original = joined._gather_ordered
    calls = []

    def observed(value, packed, lhs, rhs):
        calls.append(rhs.size)
        return original(value, packed, lhs, rhs)

    monkeypatch.setattr(joined, "_gather_ordered", observed)
    actual = joined._inner(x, ids, unit, weights, scales, h, m)
    mx.eval(actual)
    assert calls == ([rows*9, rows*9] if rows in (3, 4) and h == 512 else [])

    def native(value, packed, lhs, rhs):
        return mx.gather_qmm(value, *packed, lhs_indices=lhs, rhs_indices=rhs,
                             transpose=True, group_size=32, bits=4, mode="mxfp4", sorted_indices=True)

    monkeypatch.setattr(joined, "_gather_ordered", native)
    expected = joined._inner(x, ids, unit, weights, scales, h, m)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
