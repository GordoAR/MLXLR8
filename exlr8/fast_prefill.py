"""Expert prefill with sorted rows and direct trellis matrix multiplication.

The block table follows PonyExl3's EXL3SwitchGLU prefill path (Apache-2.0).
Weights remain packed. The Metal kernel decodes only its current 32-by-64
tile into threadgroup memory. Side vectors and both normalized Hadamard
transforms use float32, as in :mod:`exlr8.trellis`.
"""

# Block table adapted from PonyExl3 (Apache-2.0).
# Copyright 2026 Theinruj Toranavikrai. See NOTICE and third_party/licenses/.

from functools import lru_cache
from pathlib import Path

from .trellis import _check_format, _hadamard


@lru_cache(maxsize=16)
def _segment_table(experts, blocks, block_rows):
    import mlx.core as mx

    @mx.compile
    def build(sorted_ids):
        counts = mx.zeros((experts,), dtype=mx.uint32).at[sorted_ids].add(1)
        starts = mx.cumsum(counts) - counts
        block_counts = (counts + block_rows - 1) // block_rows
        block_ends = mx.cumsum(block_counts)
        marks = mx.zeros((blocks + 1,), dtype=mx.uint32).at[block_ends].add(1)
        block_experts = mx.minimum(mx.cumsum(marks)[:blocks], experts - 1)
        local = mx.arange(blocks, dtype=mx.uint32) - (block_ends - block_counts)[block_experts]
        first = starts[block_experts] + local * block_rows
        length = mx.minimum(counts[block_experts] - local * block_rows, block_rows)
        return mx.stack([block_experts, first, length]), block_ends[-1:]

    return build


def _plan(indices, experts, block_rows):
    import mlx.core as mx

    ids = indices.reshape(-1).astype(mx.uint32)
    order = mx.argsort(ids)
    sorted_ids = ids[order]
    blocks = min(ids.size, ids.size // block_rows + experts)
    table, count = _segment_table(experts, blocks, block_rows)(sorted_ids)
    return sorted_ids, order, table, count


@lru_cache(maxsize=48)
def _kernel(k, sharded_side, block_rows, gate_up=False, half_dot=False):
    import mlx.core as mx

    source = (Path(__file__).with_name("metal") / "fast_prefill.metal").read_text()
    prefix = (
        f"#define K_BITS {k}u\n"
        f"#define OUTPUT_GROUPED {int(sharded_side != 2)}\n"
        f"#define BLOCK_ROWS {block_rows}u\n"
        f"#define GATE_UP {int(gate_up)}\n"
        f"#define HALF_DOT {int(half_dot)}\n"
    )
    inputs = ["xh", "packed_gate", "packed_up", "table", "count", "dims"] if gate_up else ["xh", "packed", "table", "count", "dims"]
    return mx.fast.metal_kernel(
        name=f"exlr8_prefill_k{k}_s{sharded_side}_m{block_rows}_gu{int(gate_up)}_h{int(half_dot)}",
        input_names=inputs,
        output_names=["out"],
        source=prefix + source,
        header="#include <metal_simdgroup_matrix>\n" + (Path(__file__).with_name("metal") / "fast_decode_common.metal").read_text(),
    )


def _inner_sorted(xh, packed, k, in_features, out_features, sharded_side, table, count, block_rows, rows=None, packed_up=None, half_dot=False):
    import mlx.core as mx

    if rows is None:
        rows = xh.shape[0]
        xh = mx.concatenate([xh, mx.zeros((block_rows, in_features), dtype=mx.float32)])
    if half_dot:
        xh = xh.astype(mx.float16)
    projections = 2 if packed_up is not None else 1
    dims = mx.array([in_features // 16, out_features // 16, table.shape[1], rows, rows + block_rows], dtype=mx.uint32)
    inputs = [xh, packed.reshape(-1).view(mx.uint32)]
    if packed_up is not None:
        inputs.append(packed_up.reshape(-1).view(mx.uint32))
    inputs.extend([table, count, dims])
    return _kernel(k, sharded_side, block_rows, packed_up is not None, half_dot)(
        inputs=inputs,
        grid=(out_features // 64 * 128, table.shape[1], projections),
        threadgroup=(128, 1, 1),
        output_shapes=[(projections, rows, out_features) if packed_up is not None else (rows, out_features)],
        output_dtypes=[mx.float32],
    )[0]


def _slot_stride(x, indices, in_features):
    if x.shape[-1] != in_features:
        raise ValueError("activation width does not match in_features")
    if x.shape[:-1] == indices.shape:
        return 1
    if x.shape[:-1] == indices.shape[:-1]:
        return indices.shape[-1]
    raise ValueError("activation rows do not match the expert indices")


def _block_rows(rows, experts):
    # Small segments need fewer discarded matrix rows. Large segments use
    # larger tiles to reduce weight reads. Float32 64-row tiles were slower.
    if rows < experts * 8:
        return 8
    return 32


@lru_cache(maxsize=3)
def _transform_kernel(name):
    import mlx.core as mx

    directory = Path(__file__).with_name("metal")
    inputs = {
        "prep": ["x", "order", "indices", "in_gate", "in_up", "dims"],
        "gate_finish": ["inner", "indices", "out_gate", "out_up", "in_down", "dims"],
        "finish": ["inner", "order", "indices", "out_scale", "dims"],
    }
    return mx.fast.metal_kernel(
        name=f"exlr8_prefill_{name}", input_names=inputs[name], output_names=["out"],
        source=(directory / f"fast_prefill_{name}.metal").read_text(),
        header=(directory / "fast_decode_common.metal").read_text(),
    )


def _prep(x, ids, order, in_gate, in_up, features, slots, block_rows, projections=1):
    import mlx.core as mx

    return _transform_kernel("prep")(
        inputs=[x.reshape(-1, features), order, ids, in_gate, in_up, mx.array([features, ids.size, slots, block_rows], dtype=mx.uint32)],
        grid=(features // 128 * 32, ids.size + block_rows, projections),
        threadgroup=(128, 1, 1),
        output_shapes=[(projections, ids.size + block_rows, features) if projections == 2 else (ids.size + block_rows, features)],
        output_dtypes=[mx.float32],
    )[0]


def _finish(inner, ids, order, out_scale, features, dtype):
    import mlx.core as mx

    return _transform_kernel("finish")(
        inputs=[inner, order, ids, out_scale, mx.array([features], dtype=mx.uint32)],
        template=[("T", dtype)], grid=(features // 128 * 32, ids.size, 1),
        threadgroup=(128, 1, 1), output_shapes=[(ids.size, features)], output_dtypes=[dtype],
    )[0]


def grouped_expert_linear(x, indices, packed, in_scale, out_scale, k, in_features, out_features, sharded_side=1, *, half_dot=False):
    """Apply packed experts; ``half_dot`` changes only matrix operands.

    The experimental FP16 operands use float32 accumulation. Scales and
    both Hadamard transforms remain float32. The default is a float32 dot.
    """
    experts = _check_format(packed, k, in_features, out_features, sharded_side)
    if in_scale.shape != (experts, in_features) or out_scale.shape != (experts, out_features):
        raise ValueError("expert side vectors must contain full signed channel scales")
    slots = _slot_stride(x, indices, in_features)
    block_rows = _block_rows(indices.size, experts)
    ids, order, table, count = _plan(indices, experts, block_rows)
    xh = _prep(x, ids, order, in_scale, in_scale, in_features, slots, block_rows)
    inner = _inner_sorted(xh, packed, k, in_features, out_features, sharded_side, table, count, block_rows, rows=ids.size, half_dot=half_dot)
    return _finish(inner, ids, order, out_scale, out_features, x.dtype).reshape(indices.shape + (out_features,))


def trellis_linear_prefill(x, packed, suh, svh, k, in_features, out_features, sharded_side=1):
    """Apply one dense trellis matrix with matrix tiles and no expert sort."""
    import mlx.core as mx

    if _check_format(packed, k, in_features, out_features, sharded_side) != 1:
        raise ValueError("dense trellis requires one matrix")
    if x.shape[-1] != in_features or suh.size != in_features or svh.size != out_features:
        raise ValueError("activation or side vector dimensions do not match")
    rows = x.size // in_features
    block_rows = _block_rows(rows, 1)
    blocks = (rows + block_rows - 1) // block_rows
    first = mx.arange(blocks, dtype=mx.uint32) * block_rows
    table = mx.stack([mx.zeros((blocks,), dtype=mx.uint32), first, mx.minimum(rows - first, block_rows)])
    count = mx.array([blocks], dtype=mx.uint32)
    xh = _hadamard(x.reshape(rows, in_features).astype(mx.float32) * suh.astype(mx.float32))
    inner = _inner_sorted(xh, packed, k, in_features, out_features, sharded_side, table, count, block_rows)
    y = _hadamard(inner) * svh.astype(mx.float32)
    return y.reshape(x.shape[:-1] + (out_features,)).astype(x.dtype)


def grouped_expert_mlp(x, indices, packed, scales, k, hidden, intermediate, *, half_dot=False):
    """Apply gate/up/down projections with one expert sort and block table.

    ``packed`` and ``scales`` are the dictionaries from ``ExpertLayer``.
    The result has shape ``indices.shape + (hidden,)``.
    ``half_dot=True`` is an experiment with FP16 matrix operands and
    float32 accumulation. It does not change the scales or transforms.
    """
    import mlx.core as mx

    experts = _check_format(packed["gate"], k, hidden, intermediate, 1)
    slots = _slot_stride(x, indices, hidden)
    block_rows = _block_rows(indices.size, experts)
    ids, order, table, count = _plan(indices, experts, block_rows)
    xh = _prep(x, ids, order, scales["gate"][0], scales["up"][0], hidden, slots, block_rows, 2)
    gu = _inner_sorted(xh, packed["gate"], k, hidden, intermediate, 1, table, count, block_rows, rows=ids.size, packed_up=packed["up"], half_dot=half_dot)
    down_xh = _transform_kernel("gate_finish")(
        inputs=[gu, ids, scales["gate"][1], scales["up"][1], scales["down"][0], mx.array([intermediate, ids.size, block_rows], dtype=mx.uint32)],
        template=[("T", x.dtype)], grid=(intermediate // 128 * 32, ids.size + block_rows, 1),
        threadgroup=(128, 1, 1), output_shapes=[(ids.size + block_rows, intermediate)], output_dtypes=[mx.float32],
    )[0]
    inner = _inner_sorted(down_xh, packed["down"], k, intermediate, hidden, 2, table, count, block_rows, rows=ids.size, half_dot=half_dot)
    y = _finish(inner, ids, order, scales["down"][1], hidden, x.dtype)
    return y.reshape(indices.shape + (hidden,))
