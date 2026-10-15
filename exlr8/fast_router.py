"""Fused router for small GLM target batches."""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="exlr8_router_256_8", input_names=["logits", "bias"],
        output_names=["indices", "scores"],
        source=(Path(__file__).with_name("metal") / "fast_router.metal").read_text(),
    )


def route(logits, bias):
    rows = logits.size // 256
    return _kernel()(
        inputs=[logits.reshape(rows, 256), bias],
        grid=(32 * rows, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(*logits.shape[:-1], 9), (*logits.shape[:-1], 8)],
        output_dtypes=[mx.uint32, mx.float32],
    )


def compiled_finish(experts, gate, bias):
    from .fast_decode import compiled_expert_mlp_finish

    finish = compiled_expert_mlp_finish(experts)

    @mx.compile
    def call(x, residual):
        indices, scores = route(gate(x.astype(mx.float32)), bias)
        return finish(x, indices, scores, residual)

    return call
