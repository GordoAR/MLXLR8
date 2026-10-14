import os

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1", reason="set EXLR8_TEST_METAL=1 on a Metal host")
@pytest.mark.parametrize("zeros", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_residual_rms_keeps_native_dependent_states(zeros, rows):
    import mlx.core as mx
    from exlr8.fast_norm import residual_rms_pair

    if mx.device_info()["device_name"] != "Apple M3 Ultra":
        pytest.skip("The fused profile is verified on M3 Ultra")
    rng = np.random.default_rng(733)
    seed = rng.normal(size=(1, rows, 6144)).astype(np.float32)
    if zeros:
        seed.fill(-0.0)
    expected = actual = mx.array(seed, dtype=mx.bfloat16)
    weight = mx.array(rng.uniform(0.7, 1.3, 6144), dtype=mx.bfloat16)
    for _ in range(3):
        residual = np.zeros_like(seed) if zeros else rng.normal(size=seed.shape)
        if zeros:
            residual.fill(-0.0)
        residual = mx.array(residual, dtype=mx.bfloat16)
        h = expected + residual
        expected = mx.fast.rms_norm(h, weight, 1e-5)
        fused_h, actual = residual_rms_pair(actual, residual, weight)
        np.testing.assert_array_equal(np.array(fused_h.view(mx.uint16)), np.array(h.view(mx.uint16)))
        np.testing.assert_array_equal(np.array(actual.view(mx.uint16)), np.array(expected.view(mx.uint16)))
