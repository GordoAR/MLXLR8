"""Resident K4 experts re-encoded as native MLX affine4, in groups of 128.

The inner weights are approximate. Signed scales, Hadamard transforms,
activation casts, and the routed/shared/residual sum keep the existing path.
"""

import mlx.core as mx

from . import fast_decode as decode, fast_prefill as prefill
from .trellis import _device_codebook, _kernel


def encode(expert):
    """Convert one expert matrix at a time; do not retain decoded weights."""
    weights = {}
    h, m = expert.hidden, expert.intermediate
    for projection in ("gate", "up", "down"):
        incoming, outgoing, side = (m, h, 2) if projection == "down" else (h, m, 1)
        source = expert.packed[projection].reshape(-1)
        dims = mx.array([incoming // 16, outgoing // 16], mx.uint32)
        chunks = []
        for start in range(0, source.nbytes, expert.expert_bytes):
            packed = source[start:start + expert.expert_bytes]
            inner = _kernel(4, side, True)(
                inputs=[packed.view(mx.uint32), _device_codebook(), dims],
                grid=(incoming * outgoing, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(incoming, outgoing)], output_dtypes=[mx.float32],
            )[0]
            chunk = mx.quantize(inner.T[None], group_size=128, bits=4, mode="affine")
            mx.eval(chunk)
            chunks.append(chunk)
        weights[projection] = tuple(
            mx.concatenate([chunk[i] for chunk in chunks], axis=0) for i in range(3)
        )
        mx.eval(weights[projection])
        del chunks, packed, inner, chunk
        mx.clear_cache()
    return weights


def _inner(x, indices, weights, scales, h, m):
    slots = indices.shape[-1]
    indices = indices.reshape(-1, slots)
    x = x.reshape(-1, h)
    flat = indices.reshape(-1)
    if indices.shape[0] == 1:
        ids, order = flat, None
        xh = decode._prep(x, ids, scales["gate"][0], scales["up"][0], h, slots, False, 2)
        xh = xh.reshape(2, ids.size, h)
    else:
        order = mx.argsort(flat)
        ids = flat[order]
        xh = prefill._prep(x, ids, order, scales["gate"][0], scales["up"][0], h, slots, 8, 2)

    def dot(value, projection, incoming, outgoing):
        packed, scale, bias = weights[projection]
        return mx.gather_qmm(
            value.reshape(ids.size, 1, incoming), packed, scale, bias,
            rhs_indices=ids, transpose=True, group_size=128, bits=4,
            mode="affine", sorted_indices=(order is not None),
        ).reshape(ids.size, outgoing)

    gu = mx.stack([dot(xh[i, :ids.size], p, h, m) for i, p in enumerate(("gate", "up"))])
    if order is None:
        hidden = decode._transform_kernel("gate_finish")(
            inputs=[gu, ids, scales["gate"][1], scales["up"][1], scales["down"][0],
                    decode._dims(m, 0, ids.size, 1)], template=[("T", x.dtype)],
            grid=(m // 128 * 32, ids.size, 1), threadgroup=(128, 1, 1),
            output_shapes=[(ids.size, m)], output_dtypes=[mx.float32],
        )[0]
    else:
        hidden = prefill._transform_kernel("gate_finish")(
            inputs=[gu, ids, scales["gate"][1], scales["up"][1], scales["down"][0],
                    mx.array([m, ids.size, 8], mx.uint32)], template=[("T", x.dtype)],
            grid=(m // 128 * 32, ids.size + 8, 1), threadgroup=(128, 1, 1),
            output_shapes=[(ids.size + 8, m)], output_dtypes=[mx.float32],
        )[0]
    value = dot(hidden[:ids.size], "down", m, h)
    return value if order is None else value[mx.argsort(order)]


class NativeAffine:
    def __init__(self, expert):
        weights = self.weights = encode(expert)
        scales, h, m = expert.scales, expert.hidden, expert.intermediate

        @mx.compile
        def slots(x, ids):
            value = _inner(x, ids, weights, scales, h, m)
            return decode._finish(value, ids.reshape(-1), scales["down"][1], h, 1, x.dtype).reshape(ids.shape + (h,))

        @mx.compile
        def finish(x, ids, scores, residual):
            value = _inner(x, ids, weights, scales, h, m)
            return decode._finish_fused(
                value, ids.reshape(-1), scales["down"][1], h,
                scores.reshape(-1, 8), residual.reshape(-1, h),
            ).reshape(residual.shape)

        self.slots, self.finish = slots, finish
