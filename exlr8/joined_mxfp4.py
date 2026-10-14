"""Resident four-bit experts with folded input scales and joined gate/up."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from . import fast_decode as decode
from .trellis import _device_codebook, _kernel


def encode(expert):
    unit = mx.sign(expert.scales["gate"][0][:1])
    mx.eval(unit)
    h, m = expert.hidden, expert.intermediate
    encoded = {}
    for projection in ("gate", "up", "down"):
        incoming, outgoing, side = (m, h, 2) if projection == "down" else (h, m, 1)
        source = expert.packed[projection].reshape(-1)
        dims = mx.array([incoming // 16, outgoing // 16], mx.uint32)
        in_scales = (expert.scales[projection][0] * unit).astype(mx.float32) if side == 1 else None
        chunks = []
        for index, start in enumerate(range(0, source.nbytes, expert.expert_bytes)):
            packed = source[start:start + expert.expert_bytes]
            inner = _kernel(4, side, True)(
                inputs=[packed.view(mx.uint32), _device_codebook(), dims],
                grid=(incoming * outgoing, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(incoming, outgoing)], output_dtypes=[mx.float32],
            )[0]
            if in_scales is not None:
                inner = inner * in_scales[index, :, None]
            chunk = mx.quantize(inner.T[None], mode="mxfp4")
            mx.eval(chunk)
            chunks.append(chunk)
        encoded[projection] = tuple(
            mx.concatenate([chunk[i] for chunk in chunks], axis=0) for i in range(2)
        )
        mx.eval(encoded[projection])
        del chunks, packed, inner, chunk
        mx.clear_cache()
    weights = {
        "gate_up": tuple(mx.concatenate((encoded["gate"][i], encoded["up"][i]), axis=1)
                         for i in range(2)) + (None,),
        "down": encoded["down"] + (None,),
    }
    mx.eval(weights)
    return unit, weights


@lru_cache(maxsize=1)
def _gate_kernel():
    directory = Path(__file__).with_name("metal")
    return mx.fast.metal_kernel(
        name="exlr8_joined_gate_finish",
        input_names=["inner", "indices", "out_gate", "out_up", "in_down", "dims"],
        output_names=["out"],
        source=(directory / "joined_gate_finish.metal").read_text(),
        header=(directory / "fast_decode_common.metal").read_text(),
    )


def _inner(x, indices, unit, weights, scales, h, m):
    slots = indices.shape[-1]
    indices = indices.reshape(-1, slots)
    rows = indices.shape[0]
    x = x.reshape(rows, h)
    flat = indices.reshape(-1)
    xh = decode._prep(x, mx.zeros((rows,), mx.uint32), unit, unit, h, 1, True, 1)
    xh = xh.reshape(rows, 1, h)
    order = mx.argsort(flat) if rows > 1 else None
    ids = flat if order is None else flat[order]
    lhs = mx.arange(flat.size, dtype=mx.uint32) // slots if order is None else (order // slots).astype(mx.uint32)
    # MLX's sorted RHS GEMM requires no LHS indices and four rows per expert.
    grouped = ids.size >= 4 * weights["gate_up"][0].shape[0]
    if grouped:
        xh = xh.reshape(rows, h)[lhs].reshape(ids.size, 1, h)
    gu = mx.gather_qmm(
        xh, *weights["gate_up"], lhs_indices=None if grouped else lhs,
        rhs_indices=ids, transpose=True, group_size=32, bits=4,
        mode="mxfp4", sorted_indices=order is not None,
    ).reshape(ids.size, 2 * m)
    hidden = _gate_kernel()(
        inputs=[gu, ids, scales["gate"][1], scales["up"][1], scales["down"][0],
                decode._dims(m, 0, ids.size, 1)],
        template=[("T", x.dtype)], grid=(m // 128 * 32, ids.size, 1),
        threadgroup=(128, 1, 1), output_shapes=[(ids.size, m)], output_dtypes=[mx.float32],
    )[0]
    value = mx.gather_qmm(
        hidden.reshape(ids.size, 1, m), *weights["down"],
        lhs_indices=None if grouped else mx.arange(ids.size, dtype=mx.uint32),
        rhs_indices=ids, transpose=True, group_size=32, bits=4,
        mode="mxfp4", sorted_indices=order is not None,
    ).reshape(ids.size, h)
    return value if order is None else value[mx.argsort(order)]


class JoinedMXFP4:
    def __init__(self, expert):
        unit, weights = encode(expert)
        self.unit, self.weights = unit, weights
        scales, h, m = expert.scales, expert.hidden, expert.intermediate

        @mx.compile
        def slots(x, ids):
            value = _inner(x, ids, unit, weights, scales, h, m)
            return decode._finish(value, ids.reshape(-1), scales["down"][1], h, 1, x.dtype).reshape(ids.shape + (h,))

        @mx.compile
        def finish(x, ids, scores, residual):
            value = _inner(x, ids, unit, weights, scales, h, m)
            return decode._finish_fused(
                value, ids.reshape(-1), scales["down"][1], h,
                scores.reshape(-1, 8), residual.reshape(-1, h),
            ).reshape(residual.shape)

        self.slots, self.finish = slots, finish
