"""Lossless E4M3 expansion for the small dense trellis matrices."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .trellis import _check_format, _device_codebook, _hadamard


@lru_cache(maxsize=6)
def _kernel(k, sharded_side):
    source = (Path(__file__).with_name("metal") / "dense_fp8_decode.metal").read_text()
    return mx.fast.metal_kernel(
        name=f"exlr8_dense_fp8_k{k}_s{sharded_side}",
        input_names=["packed", "codebook", "dims"], output_names=["out"],
        source=f"#define K_BITS {k}u\n#define OUTPUT_GROUPED {int(sharded_side != 2)}\n" + source,
    )


def expand_trellis(packed, k, in_features, out_features, sharded_side=1):
    """Return the exact inner E4M3 codes in native MXFP8 matrix order."""
    if _check_format(packed, k, in_features, out_features, sharded_side) != 1:
        raise ValueError("dense expansion requires one matrix")
    dims = mx.array([in_features // 16, out_features // 16], dtype=mx.uint32)
    codes = _kernel(k, sharded_side)(
        inputs=[packed.reshape(-1).view(mx.uint32), _device_codebook(), dims],
        grid=(in_features * out_features // 4, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(out_features, in_features // 4)], output_dtypes=[mx.uint32],
    )[0]
    return codes, mx.full((out_features, in_features // 32), 127, dtype=mx.uint8)


@mx.compile
def expanded_linear(x, packed, exponents, suh, svh):
    """Keep both F32 side scales and the two normalized block transforms."""
    xh = _hadamard(x.astype(mx.float32) * suh.astype(mx.float32))
    inner = mx.quantized_matmul(xh, packed, scales=exponents, mode="mxfp8")
    return (_hadamard(inner) * svh.astype(mx.float32)).astype(x.dtype)


@lru_cache(maxsize=2)
def _k4_kernel(k, n):
    assert (k, n) in ((2048, 16384), (16384, 6144))
    side = 1 if k == 2048 else 2
    directory = Path(__file__).with_name("metal")
    body, source = (directory / "dense_k4_decode.metal").read_text().split("// KERNEL_ENTRY\n", 1)
    if side == 2:
        unit = "U s = dequantize_scale<U, group_size>(sl[0]);"
        assert body.count(unit) == 1
        body = body.replace(unit, "U s = U(1);")
    header = (f"constant constexpr int INPUT_K = {k};\n"
              f"constant constexpr int OUTPUT_N = {n};\n"
              f"constant constexpr int K4_SIDE = {side};\n")
    header += (directory / "fast_decode_common.metal").read_text()
    header += body
    return mx.fast.metal_kernel(name=f"exlr8_dense_k4_k{k}_n{n}",
        input_names=["xh", "weights", "scales"], output_names=["out"],
        header=header, source=source)


@mx.compile
def k4_linear(x, packed_k4, exponents, suh, svh):
    # DenseLinear selects only BF16 shape (1, 1, K) and the two tested records.
    k, n = x.shape[-1], exponents.shape[0]
    xh = _hadamard(x.astype(mx.float32) * suh.astype(mx.float32))
    out = _k4_kernel(k, n)(inputs=[xh.reshape(1, k), packed_k4, exponents],
        grid=(32, n // 8 * 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(1, n)], output_dtypes=[mx.float32])[0].reshape(1, 1, n)
    return (_hadamard(out) * svh.astype(mx.float32)).astype(x.dtype)
