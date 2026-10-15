"""Packed FP8 vocabulary projection for one or four BF16 token rows."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=8)
def _kernel(rows, k, n):
    header, body = (Path(__file__).with_name("metal") / "vocab_fp8.metal").read_text().split("// KERNEL_ENTRY\n", 1)
    header = f"#define BATCH_ROWS {rows}\nconstant constexpr uint INPUT_K={k}, OUTPUT_N={n};\n" + header
    return mx.fast.metal_kernel(name=f"exlr8_vocab_fp8_r{rows}_k{k}_n{n}",
        input_names=["xh", "weights", "row_scales"], output_names=["out"],
        header=header, source=body)


@mx.compile
def project(x, packed, scales):
    rows, k, n = x.shape[1], x.shape[2], scales.size
    assert rows in (1, 4) and k % 256 == 0 and n % 8 == 0
    return _kernel(rows, k, n)(inputs=[x.reshape(rows, k).astype(mx.float32), packed, scales],
        template=[("T", x.dtype)], grid=(32, n // 8 * 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(1, rows, n)], output_dtypes=[x.dtype])[0]
