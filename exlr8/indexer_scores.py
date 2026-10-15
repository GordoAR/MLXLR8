"""Fused F32 index scores for one-to-four-row BF16 queries."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    source = (Path(__file__).with_name("metal") / "indexer_scores.metal").read_text()
    return mx.fast.metal_kernel(name="exlr8_indexer_scores",
        input_names=["q", "k", "weights", "valid", "dims"], output_names=["out"],
        header="#include <metal_simdgroup_matrix>\n", source=source,
        compile_options={"math_mode": "safe"})


def score(q, k, weights, mask):
    rows, keys = q.shape[2], k.shape[2]
    valid = (mx.ones((rows, keys), mx.bool_) if mask is None else
             mx.broadcast_to(mask, (1, 1, rows, keys)).reshape(rows, keys))
    return _kernel()(inputs=[q, k, weights, valid, mx.array([keys], mx.uint32)],
        template=[("T", mx.bfloat16), ("ROWS", rows), ("BN", 32)],
        grid=(((keys + 31) // 32) * 128, 1, rows), threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, rows, keys)], output_dtypes=[mx.float32])[0]
