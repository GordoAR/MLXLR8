"""Residual-add/RMS pair for the verified small-batch BF16 kernel profile."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name="exlr8_residual_rms_q1", input_names=["x", "residual", "w", "epsilon"],
        output_names=["residual_out", "out"],
        source=(Path(__file__).with_name("metal") / "residual_rms.metal").read_text(),
        compile_options={"math_mode": "safe"},
    )
    return kernel, mx.array([1e-5], dtype=mx.float32)


def residual_rms_pair(x, residual, weight):
    import mlx.core as mx

    kernel, epsilon = _kernel()
    return kernel(
        inputs=[x, residual, weight, epsilon], template=[("T", mx.bfloat16)],
        grid=(1024 * x.shape[1], 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[x.shape, x.shape], output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
