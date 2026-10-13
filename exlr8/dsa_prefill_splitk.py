"""Private batched split-K PV experiment. The runtime does not import it.

Use four fixed partitions, F32 matrix operands and partial sums, then one
BF16 output cast. BF16 probabilities are unchanged. No F16 underflow occurs.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .dsa_prefill import _gathered_rows


@lru_cache(maxsize=1)
def _pv_kernel():
    source = (Path(__file__).with_name("metal") / "dsa_prefill_splitk_pv.metal").read_text()
    return mx.fast.metal_kernel(name="exlr8_dsa_splitk_pv",
                                input_names=["p", "v"], output_names=["partial"], source=source)


def splitk_pv(probabilities, values):
    """Compute P@V with the same partition layout for Q1/Q2/Q4."""
    m, k, n = probabilities.shape[-2], probabilities.shape[-1], values.shape[-1]
    if probabilities.dtype != mx.bfloat16 or values.dtype != mx.bfloat16 or k % 64:
        raise ValueError("split-K PV requires BF16 operands and K divisible by 64")
    if probabilities.shape[:-2] != values.shape[:-2] or k != values.shape[-2]:
        raise ValueError("split-K PV requires equal matrix batches and matching K")
    p, v = probabilities.reshape(-1, m, k), values.reshape(-1, k, n)
    partial = _pv_kernel()(
        inputs=[p, v], template=[("M", m), ("K", k), ("N", n)],
        grid=(((n + 31) // 32) * 128, (m + 31) // 32, p.shape[0] * 4),
        threadgroup=(128, 1, 1), output_shapes=[(p.shape[0], 4, m, n)],
        output_dtypes=[mx.float32],
    )[0]
    # Use the stock split-K accumulation order: 0 + partial0 + ... + partial3.
    output = ((partial[:, 0] + partial[:, 1]) + partial[:, 2]) + partial[:, 3]
    return output.astype(mx.bfloat16).reshape(*probabilities.shape[:-1], n)


def native_splitk_pv(probabilities, values):
    """Small MLX-only control with four F32 partial products."""
    m, k, n = probabilities.shape[-2], probabilities.shape[-1], values.shape[-1]
    p = probabilities.reshape(-1, m, 4, k // 4).swapaxes(1, 2).astype(mx.float32)
    v = values.reshape(-1, 4, k // 4, n).astype(mx.float32)
    partial = p @ v
    output = ((partial[:, 0] + partial[:, 1]) + partial[:, 2]) + partial[:, 3]
    return output.astype(probabilities.dtype).reshape(*probabilities.shape[:-1], n)


def splitk_attention(q, q_pe, latent, k_pe, indices, mask, scale, chunk=32, pv=splitk_pv):
    """Keep stock BF16 QK, score-add and precise softmax boundaries."""
    batch, heads = q.shape[:2]
    outputs = []
    for qr, lr, bias, rows in _gathered_rows(q, q_pe, latent, k_pe, indices, mask, scale, chunk):
        qr, bias = qr.swapaxes(1, 2), bias.swapaxes(1, 2)
        score = (qr * scale) @ lr.swapaxes(-1, -2)
        p = mx.softmax(score + bias, axis=-1, precise=True)
        output = pv(p, lr)
        outputs.append(output.reshape(batch, rows, heads, -1).swapaxes(1, 2))
    return mx.concatenate(outputs, axis=2)


def install_splitk_control(attention):
    """Install the private Q2–Q4 control on one attention instance."""
    original_queries = attention._absorbed_queries
    original_sparse = attention._sparse_attention

    def queries(q_nope):
        length = q_nope.shape[2]
        if 2 <= length <= 4:
            return mx.concatenate(
                [original_queries(q_nope[:, :, i:i + 1]) for i in range(length)], axis=2,
            )
        return original_queries(q_nope)

    def sparse(q, q_pe, latent, k_pe, indices, mask):
        if (2 <= q.shape[2] <= 4 and indices.shape[-1] % 64 == 0
                and all(value.dtype == mx.bfloat16 for value in (q, q_pe, latent, k_pe))):
            return splitk_attention(q, q_pe, latent, k_pe, indices, mask, attention.scale)
        return original_sparse(q, q_pe, latent, k_pe, indices, mask)

    attention._absorbed_queries = queries
    attention._sparse_attention = sparse
