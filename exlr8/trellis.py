"""EXLR8 E4M3 trellis operators for MLX.

The bitstream and tile permutation follow exllamav3 0.0.43 (MIT).
The fused reduction follows PonyExl3's Metal GEMV (Apache-2.0).
The EXLR8 codebook rounds the mcg half result to finite E4M3 before either
scale. The kernel adaptations and their source notices are in metal/.

Tiles are grouped into 128-by-128 blocks. Side 1 stores output blocks
first; side 2 stores input blocks first. Inside each block the input tile
comes before the output tile. The raw half shape does not expose this order.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math

import numpy as np


def round_e4m3(values):
    """Round finite values to E4M3, with even ties and finite saturation."""
    values = np.asarray(values, dtype=np.float32)
    codes = np.arange(127, dtype=np.int32)
    exponent, fraction = codes >> 3, codes & 7
    levels = np.where(
        exponent == 0,
        fraction * 2.0**-9,
        (8 + fraction) * np.exp2(exponent - 10),
    ).astype(np.float32)
    magnitude = np.minimum(np.abs(values), 448.0)
    upper = np.minimum(np.searchsorted(levels, magnitude), 126)
    lower = np.maximum(upper - 1, 0)
    high_distance = levels[upper] - magnitude
    low_distance = magnitude - levels[lower]
    take_upper = (high_distance < low_distance) | (
        (high_distance == low_distance) & ((upper & 1) == 0)
    )
    return np.copysign(levels[np.where(take_upper, upper, lower)], values)


@lru_cache(maxsize=1)
def mcg_codebook():
    """All 65536 codewords, including the encoder's mandatory E4M3 rounding."""
    words = np.arange(65536, dtype=np.uint32) * np.uint32(0xCBAC1FED)
    halves = (words & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    low = (halves & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (halves >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return round_e4m3(low + high).astype(np.float16)


def _check_format(packed, k, in_features, out_features, sharded_side):
    if k not in (2, 3, 4):
        raise ValueError("EXLR8 trellis width must be 2, 3, or 4")
    if in_features <= 0 or out_features <= 0 or in_features % 128 or out_features % 128:
        raise ValueError("trellis dimensions must be positive multiples of 128")
    if sharded_side not in (0, 1, 2):
        raise ValueError("sharded_side must be 0, 1, or 2")
    size = in_features * out_features * k // 8
    nbytes = packed.size * packed.dtype.itemsize if isinstance(packed, np.ndarray) else packed.nbytes
    if nbytes == 0 or nbytes % size:
        raise ValueError("packed byte count does not match the trellis dimensions")
    return nbytes // size


def hadamard_reference(x):
    """Normalized Sylvester transform in independent 128-channel blocks."""
    x = np.asarray(x, dtype=np.float32)
    y = x.reshape(-1, 128).copy()
    for bit in (1, 2, 4, 8, 16, 32, 64):
        pairs = y.reshape(-1, 128 // (2 * bit), 2, bit)
        a, b = pairs[:, :, 0].copy(), pairs[:, :, 1].copy()
        pairs[:, :, 0], pairs[:, :, 1] = a + b, a - b
    return (y * np.float32(1 / math.sqrt(128))).reshape(x.shape)


def decode_trellis_reference(packed, k, in_features, out_features, sharded_side=1):
    """Decode one stored matrix to inner weights [in, out], for format tests."""
    packed = np.asarray(packed)
    if _check_format(packed, k, in_features, out_features, sharded_side) != 1:
        raise ValueError("reference decode requires one matrix")
    it, ot = in_features // 16, out_features // 16
    words = np.ascontiguousarray(packed).view(np.uint32)
    if sharded_side == 2:
        words = words.reshape(it // 8, ot // 8, 8, 8, 8 * k).transpose(0, 2, 1, 3, 4)
        words = words.reshape(it, ot, 8 * k)
    else:
        words = words.reshape(ot // 8, it, 8, 8 * k).transpose(1, 0, 2, 3)
        words = words.reshape(it, ot, 8 * k)
    index = np.arange(256, dtype=np.int32)
    end = (index + 1) * k + 256 * k
    last = (end - 1) // 32
    merged = (words[..., (last - 1) % (8 * k)].astype(np.uint64) << np.uint64(32))
    merged |= words[..., last % (8 * k)].astype(np.uint64)
    codewords = ((merged >> ((last + 1) * 32 - end).astype(np.uint64)) & 65535).astype(np.uint16)
    lane, slot = index // 8, index % 8
    row = (lane % 4) * 2 + (slot & 1) + ((slot & 2) * 4)
    col = lane // 4 + ((slot & 4) * 2)
    inverse = np.argsort(row * 16 + col)
    tiles = mcg_codebook()[codewords][..., inverse].reshape(it, ot, 16, 16)
    return tiles.transpose(0, 2, 1, 3).reshape(in_features, out_features)


def dequantize_reference(packed, suh, svh, k, in_features, out_features, sharded_side=1):
    """Public matrix [out, in]; this function is for small CPU checks."""
    inner = decode_trellis_reference(packed, k, in_features, out_features, sharded_side)
    weight = hadamard_reference(hadamard_reference(inner).T)
    return weight * np.asarray(svh, np.float32)[:, None] * np.asarray(suh, np.float32)[None, :]


@lru_cache(maxsize=1)
def _device_codebook():
    import mlx.core as mx

    return mx.array(mcg_codebook())


@lru_cache(maxsize=12)
def _kernel(k, sharded_side, decode=False):
    import mlx.core as mx

    name = "decode" if decode else "trellis"
    source = (Path(__file__).with_name("metal") / f"{name}.metal").read_text()
    prefix = f"#define K_BITS {k}u\n#define OUTPUT_GROUPED {int(sharded_side != 2)}\n"
    return mx.fast.metal_kernel(
        name=f"exlr8_{name}_k{k}_s{sharded_side}",
        input_names=["packed", "codebook", "dims"] if decode else ["xh", "packed", "codebook", "indices", "dims"],
        output_names=["out"],
        source=prefix + source,
    )


def _hadamard(x):
    import mlx.core as mx

    return mx.hadamard_transform(x.reshape(-1, x.shape[-1] // 128, 128), scale=1 / math.sqrt(128)).reshape(x.shape)


def _inner_linear(xh, indices, packed, k, in_features, out_features, sharded_side):
    import mlx.core as mx

    rows = xh.size // in_features
    it, ot = in_features // 16, out_features // 16
    splits = min(8, max(1, (it + 63) // 64))
    dims = mx.array([it, ot, rows, splits], dtype=mx.uint32)
    y = _kernel(k, sharded_side)(
        inputs=[xh.reshape(rows, in_features), packed.reshape(-1).view(mx.uint32), _device_codebook(), indices.reshape(-1).astype(mx.uint32), dims],
        grid=(ot * 128, rows, splits),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, splits, out_features)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.sum(y, axis=1)


def trellis_linear(x, packed, suh, svh, k, in_features, out_features, sharded_side=1):
    """Apply a dense stored trellis matrix without creating decoded weights."""
    import mlx.core as mx

    if _check_format(packed, k, in_features, out_features, sharded_side) != 1:
        raise ValueError("dense trellis requires one matrix")
    if x.shape[-1] != in_features or suh.size != in_features or svh.size != out_features:
        raise ValueError("activation or side vector dimensions do not match")
    xh = _hadamard(x.astype(mx.float32) * suh.astype(mx.float32))
    indices = mx.zeros((xh.size // in_features,), dtype=mx.uint32)
    inner = _inner_linear(xh, indices, packed, k, in_features, out_features, sharded_side)
    y = _hadamard(inner) * svh.astype(mx.float32)
    return y.reshape(x.shape[:-1] + (out_features,)).astype(x.dtype)


def expert_linear(x, indices, packed, in_scale, out_scale, k, in_features, out_features, sharded_side=1):
    """Apply selected experts; return [*indices.shape, out_features].

    ``packed`` holds all experts or a compact, reindexed set. Side vectors
    have shapes [experts, in] and [experts, out], with signs included.
    Input has one row per token or one row per expert slot.
    """
    import mlx.core as mx

    experts = _check_format(packed, k, in_features, out_features, sharded_side)
    if x.shape[-1] != in_features:
        raise ValueError("activation width does not match in_features")
    if in_scale.shape != (experts, in_features) or out_scale.shape != (experts, out_features):
        raise ValueError("expert side vectors must contain full signed channel scales")
    if x.shape[:-1] == indices.shape:
        rows = x
    elif x.shape[:-1] == indices.shape[:-1]:
        rows = mx.broadcast_to(x[..., None, :], indices.shape + (in_features,))
    else:
        raise ValueError("activation rows do not match the expert indices")
    ids = indices.reshape(-1).astype(mx.uint32)
    xh = _hadamard(rows.reshape(-1, in_features).astype(mx.float32) * in_scale[ids].astype(mx.float32))
    inner = _inner_linear(xh, ids, packed, k, in_features, out_features, sharded_side)
    y = _hadamard(inner) * out_scale[ids].astype(mx.float32)
    return y.reshape(indices.shape + (out_features,)).astype(x.dtype)


def dequantize_trellis(packed, suh, svh, k, in_features, out_features, sharded_side=1):
    """Decode one dense public matrix [out, in] for inspection."""
    import mlx.core as mx

    if _check_format(packed, k, in_features, out_features, sharded_side) != 1:
        raise ValueError("dense dequantization requires one matrix")
    dims = mx.array([in_features // 16, out_features // 16], dtype=mx.uint32)
    inner = _kernel(k, sharded_side, True)(
        inputs=[packed.reshape(-1).view(mx.uint32), _device_codebook(), dims],
        grid=(in_features * out_features, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(in_features, out_features)],
        output_dtypes=[mx.float32],
    )[0]
    weight = _hadamard(_hadamard(inner).T)
    return weight * svh.astype(mx.float32)[:, None] * suh.astype(mx.float32)[None, :]
