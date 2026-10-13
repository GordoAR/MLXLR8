"""Opt-in sparse attention with half matrix operands and float accumulators.

BF16 values in the F16 normal range convert without error. F16 also holds
all BF16 values with magnitudes at least 2**-17, if subnormals are retained.
Probabilities below that limit can round. Values at or below 2**-25 round
to zero. The Metal tests check subnormal operands separately.
"""

from functools import lru_cache
from pathlib import Path

from .dsa_prefill import _gathered_rows


@lru_cache(maxsize=2)
def _kernel(transpose_b):
    import mlx.core as mx

    source = (Path(__file__).with_name("metal") / "dsa_prefill_half.metal").read_text()
    return mx.fast.metal_kernel(
        name=f"exlr8_dsa_half_gemm_t{int(transpose_b)}",
        input_names=["a", "b", "dims"], output_names=["out"],
        source=f"#define TRANSPOSE_B {int(transpose_b)}\n" + source,
        header="#include <metal_simdgroup_matrix>\n",
    )


@lru_cache(maxsize=32)
def _dims(m, n, k):
    import mlx.core as mx

    return mx.array([m, n, k], dtype=mx.uint32)


def half_gemm(a, b, *, transpose_b=False, output_dtype=None):
    """Multiply matching 3D batches with half operands and F32 accumulation."""
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.dtype != b.dtype:
        raise ValueError("half GEMM needs matching 3D batches and storage types")
    batch, m, k = a.shape
    n, inner = b.shape[-2:] if transpose_b else b.shape[-2:][::-1]
    if k != inner:
        raise ValueError("half GEMM inner dimensions do not match")
    dtype = output_dtype or a.dtype
    return _kernel(transpose_b)(
        inputs=[a, b, _dims(m, n, k)], template=[("T", a.dtype), ("U", dtype)],
        grid=(((n + 63) // 64) * 128, (m + 63) // 64, batch),
        threadgroup=(128, 1, 1), output_shapes=[(batch, m, n)],
        output_dtypes=[dtype],
    )[0]


def half_heads_as_rows_attention(q, q_pe, latent, k_pe, indices, mask, scale,
                                 *, half_qk=True, half_pv=True, chunk=32):
    """Keep stock dtype boundaries and replace selected attention GEMMs."""
    import mlx.core as mx

    batch, heads = q.shape[:2]
    outputs = []
    for qr, lr, bias, rows in _gathered_rows(q, q_pe, latent, k_pe, indices, mask, scale, chunk):
        scaled = qr[:, :, 0, :] * scale
        values = lr[:, 0, :, :]
        scores = (half_gemm(scaled, values, transpose_b=True) if half_qk
                  else scaled @ values.swapaxes(-1, -2))
        scores = scores + bias[:, :, 0, :]
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        output = (half_gemm(probabilities, values) if half_pv
                  else probabilities @ values)
        outputs.append(output.reshape(batch, rows, heads, -1).swapaxes(1, 2))
    return mx.concatenate(outputs, axis=2)
