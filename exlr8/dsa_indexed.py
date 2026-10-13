"""Indexed BF16 DSA, enabled by the optimized model loader.

The fixed tiles match MLX 0.32.3 Steel on M3 Ultra for native 32-query
chunks: QK 64x64x16, PV 32x64x16, each with 1x2 SIMD groups. Smaller
native chunks can use different tiles. Keep their final row tail native.
Both products use F32 operands and accumulation with a BF16 result.
Q1 fuses native 64x32x32 QK/PE tiles and uses native 32-head PV tiles
with four ordered 512-key F32 partitions and one final BF16 cast.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=2)
def _kernel(product):
    source = (Path(__file__).with_name("metal") / f"dsa_indexed_{product}.metal").read_text()
    names = ["q", "latent", "ids"] if product == "qk" else ["p", "v", "ids"]
    return mx.fast.metal_kernel(name=f"exlr8_dsa_indexed_{product}",
        input_names=names, output_names=["out"], source=source,
        header="#include <metal_simdgroup_matrix>\n")


def _qk(scaled_q, latent, ids):
    rows = scaled_q.shape[0]
    return _kernel("qk")(inputs=[scaled_q, latent, ids], template=[("T", mx.bfloat16)],
        grid=(32 * 64, 1, rows), threadgroup=(64, 1, 1),
        output_shapes=[(rows, 64, 2048)], output_dtypes=[mx.bfloat16])[0]


def _pv(probabilities, latent, ids):
    rows = probabilities.shape[0]
    return _kernel("pv")(inputs=[probabilities, latent, ids], template=[("T", mx.bfloat16)],
        grid=(8 * 64, 2, rows), threadgroup=(64, 1, 1),
        output_shapes=[(rows, 64, 512)], output_dtypes=[mx.bfloat16])[0]


@lru_cache(maxsize=3)
def _decode_kernel(stage):
    source = (Path(__file__).with_name("metal") / "dsa_indexed_decode.metal").read_text()
    names = (["q", "qp", "latent", "pe", "ids", "valid"] if stage == 0 else
             ["p", "v", "ids"] if stage == 1 else ["partial"])
    return mx.fast.metal_kernel(name=f"exlr8_dsa_indexed_decode_{stage}",
        input_names=names, output_names=["out"], source=f"#define STAGE {stage}\n" + source,
        header="#include <metal_simdgroup_matrix>\n")


def _decode_scores(scaled_q, scaled_qp, latent, pe, ids, valid):
    return _decode_kernel(0)(inputs=[scaled_q, scaled_qp, latent, pe, ids, valid],
        template=[("T", mx.bfloat16)], grid=(64 * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 64, 2048)], output_dtypes=[mx.bfloat16])[0]


def _decode_pv(probabilities, latent, ids):
    partial = _decode_kernel(1)(inputs=[probabilities, latent, ids], template=[("T", mx.bfloat16)],
        grid=(16 * 128, 2, 4), threadgroup=(128, 1, 1),
        output_shapes=[(1, 4, 64, 512)], output_dtypes=[mx.float32])[0]
    return _decode_kernel(2)(inputs=[partial], template=[("T", mx.bfloat16)],
        grid=(32768, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, 64, 512)], output_dtypes=[mx.bfloat16])[0]


def _indexed_decode(q, q_pe, latent, k_pe, indices, mask, scale):
    keys = latent.shape[2]
    ids = indices.reshape(1, 2048)
    valid = ids <= keys - 1
    if mask is not None:
        row_mask = mx.broadcast_to(mask, (1, 1, 1, keys)).reshape(1, keys)
        valid &= mx.take_along_axis(row_mask, ids, axis=-1)
    values, pe = latent.reshape(keys, 512), k_pe.reshape(keys, 64)
    score = _decode_scores(q.reshape(1, 64, 512) * scale,
                          q_pe.reshape(1, 64, 64) * scale, values, pe, ids, valid)
    probabilities = mx.softmax(score, axis=-1, precise=True)
    return _decode_pv(probabilities, values, ids).reshape(1, 64, 1, 512)


def _supported(q, q_pe, latent, k_pe, indices, mask):
    if len(q.shape) != 4 or len(latent.shape) != 4 or indices is None:
        return False
    length, keys = q.shape[2], latent.shape[2]
    return ((length == 1 or length >= 32) and keys >= max(2048, length)
        and q.shape == (1, 64, length, 512) and q_pe.shape == (1, 64, length, 64)
        and latent.shape == (1, 1, keys, 512) and k_pe.shape == (1, 1, keys, 64)
        and indices.shape == (1, 1, length, 2048) and indices.dtype == mx.uint32
        and all(value.dtype == mx.bfloat16 for value in (q, q_pe, latent, k_pe))
        and (mask is None or mask.dtype == mx.bool_))


def _indexed_attention(q, q_pe, latent, k_pe, indices, mask, scale, native):
    length, keys = q.shape[2], latent.shape[2]
    prefix = length - length % 32
    values, pe = latent.reshape(keys, 512), k_pe.reshape(keys, 64)
    row_mask = None if mask is None else mx.broadcast_to(mask, (1, 1, length, keys))[:, 0]
    outputs = []
    for start in range(0, prefix, 512):
        end = min(start + 512, prefix)
        rows = end - start
        ids = indices[0, 0, start:end]
        qr = q[:, :, start:end].swapaxes(1, 2).reshape(rows, 64, 512)
        qp = q_pe[:, :, start:end].swapaxes(1, 2).reshape(rows, 64, 64)
        gathered_pe = mx.take_along_axis(mx.broadcast_to(pe[None], (rows, keys, 64)),
            mx.broadcast_to(ids[..., None], (rows, 2048, 64)), axis=1)
        # Retain the native M1 PE dot and each BF16 boundary before softmax.
        bias = ((qp[:, :, None, :] * scale) @ gathered_pe[:, None].swapaxes(-1, -2)).reshape(rows, 64, 2048)
        valid = ids <= mx.arange(keys - length + start, keys - length + end)[:, None]
        if row_mask is not None:
            valid &= mx.take_along_axis(row_mask[0, start:end], ids, axis=-1)
        bias = mx.where(valid[:, None, :], bias, -float("inf"))
        score = _qk(qr * scale, values, ids)
        probabilities = mx.softmax(score + bias, axis=-1, precise=True)
        output = _pv(probabilities, values, ids)
        outputs.append(output.reshape(1, rows, 64, 512).swapaxes(1, 2))
    if prefix < length:
        tail_mask = None if row_mask is None else row_mask[:, None, prefix:]
        outputs.append(native(q[:, :, prefix:], q_pe[:, :, prefix:], latent, k_pe,
                              indices[:, :, prefix:], tail_mask))
    return mx.concatenate(outputs, axis=2)


def install_indexed_prefill(attention):
    """Wrap one instance; Q2–Q31 and unsupported inputs keep its native path."""
    native = attention._sparse_attention

    def sparse(q, q_pe, latent, k_pe, indices, mask):
        if _supported(q, q_pe, latent, k_pe, indices, mask):
            if q.shape[2] == 1:
                return _indexed_decode(q, q_pe, latent, k_pe, indices, mask, attention.scale)
            return _indexed_attention(q, q_pe, latent, k_pe, indices, mask, attention.scale, native)
        return native(q, q_pe, latent, k_pe, indices, mask)

    attention._sparse_attention = sparse
