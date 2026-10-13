"""Selected expert GEMV with exact inline EXLR8 codebook decoding."""

from functools import lru_cache
from pathlib import Path

from .trellis import _check_format


@lru_cache(maxsize=None)
def _kernel(k, side, nsg=4, quad=True, gate_up=False, direct=True, vector=True, lookup=False, reuse=1, hot=False, direct_x=False, fixed=False):
    import mlx.core as mx

    directory = Path(__file__).with_name("metal")
    prefix = (f"#define K_BITS {k}u\n#define OUTPUT_GROUPED {int(side != 2)}\n"
              f"#define NSG {nsg}u\n#define QUAD_REDUCE {int(quad)}\n"
              f"#define GATE_UP {int(gate_up)}\n#define DIRECT_K4 {int(direct)}\n"
              f"#define VECTOR_DECODE {int(vector)}\n#define LOOKUP {int(lookup)}\n"
              f"#define REUSE_ROWS {reuse}u\n#define HOT_CACHE {int(hot)}\n"
              f"#define DIRECT_X {int(direct_x and k == 4 and direct)}\n")
    names = ["xh", "packed_gate", "packed_up", "indices", "dims"] if gate_up else ["xh", "packed", "indices", "dims"]
    if lookup:
        names.append("codebook")
    header = (directory / "fast_decode_common.metal").read_text()
    if direct_x:
        header += (directory / "fast_decode_activation.metal").read_text()
    if hot:
        names.extend(["cache_gate", "cache_up", "expert_to_slot"] if gate_up else ["cache", "expert_to_slot"])
        header += (directory / "fast_decode_fp8.metal").read_text()
    source = (directory / "fast_decode_linear.metal").read_text()
    if fixed:
        dimensions = (384, 128, 9, 8) if side == 1 else (128, 384, 9, 1)
        for index, value in enumerate(dimensions):
            source = source.replace(f"dims[{index}]", f"{value}u")
    return mx.fast.metal_kernel(
        name=f"exlr8_fast_linear_k{k}_s{side}_n{nsg}_q{int(quad)}_gu{int(gate_up)}_d{int(direct)}_v{int(vector)}_l{int(lookup)}_r{reuse}_h{int(hot)}" + ("_x1" if direct_x else "") + ("_fd1" if fixed else ""),
        input_names=names,
        output_names=["out"],
        source=prefix + source,
        header=header,
    )


@lru_cache(maxsize=None)
def _transform_kernel(name, per_slot=False):
    import mlx.core as mx

    directory = Path(__file__).with_name("metal")
    inputs = {
        "prep": ["x", "indices", "in_gate", "in_up", "dims"],
        "gate_finish": ["inner", "indices", "out_gate", "out_up", "in_down", "dims"],
        "finish": ["inner", "indices", "out_scale", "dims"],
        "finish_fused": ["inner", "indices", "out_scale", "scores", "residual", "dims"],
    }
    return mx.fast.metal_kernel(
        name=f"exlr8_fast_{name}_ps{int(per_slot)}",
        input_names=inputs[name],
        output_names=["out"],
        source=f"#define PER_SLOT {int(per_slot)}\n" + (directory / f"fast_decode_{name}.metal").read_text(),
        header=(directory / "fast_decode_common.metal").read_text(),
    )


@lru_cache(maxsize=None)
def _dims(*values):
    import mlx.core as mx

    return mx.array(values, dtype=mx.uint32)


def _partials(xh, ids, packed, k, in_features, out_features, side, splits, nsg=4, quad=True, packed_up=None, direct=True, vector=True, lookup=False, reuse=1, fp8=None, expert_to_slot=None, fp8_up=None, direct_x=False):
    import mlx.core as mx

    rows = ids.size
    projections = 2 if packed_up is not None else 1
    inputs = [xh.reshape(projections * rows, in_features), packed.reshape(-1).view(mx.uint32)]
    if packed_up is not None:
        inputs.append(packed_up.reshape(-1).view(mx.uint32))
    inputs.extend([ids.reshape(-1).astype(mx.uint32), _dims(in_features // 16, out_features // 16, rows, splits)])
    if lookup:
        from .trellis import _device_codebook

        inputs.append(_device_codebook())
    if fp8 is not None:
        inputs.append(fp8.reshape(-1).view(mx.uint32))
        if packed_up is not None:
            inputs.append(fp8_up.reshape(-1).view(mx.uint32))
        inputs.append(expert_to_slot)
    fixed = (k == 4 and side in (1, 2) and nsg == 4 and quad and direct and vector
             and not lookup and reuse == 1 and fp8 is None and direct_x
             and (packed_up is not None) == (side == 1)
             and (in_features, out_features, rows, splits) ==
             ((6144, 2048, 9, 8) if side == 1 else (2048, 6144, 9, 1)))
    return _kernel(k, side, nsg, quad, packed_up is not None, direct, vector, lookup, reuse, fp8 is not None, direct_x, fixed)(
        inputs=inputs,
        grid=(out_features // 16 * nsg * 32, projections * rows, splits),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(projections * rows, splits, out_features)],
        output_dtypes=[mx.float32],
    )[0]


def _inner_linear(xh, ids, packed, k, in_features, out_features, side, splits=None, nsg=4, quad=True, direct=True, vector=True, lookup=False):
    import mlx.core as mx

    splits = splits or min(8, max(1, (in_features // 16 + 63) // 64))
    result = _partials(xh, ids, packed, k, in_features, out_features, side, splits, nsg, quad, direct=direct, vector=vector, lookup=lookup)
    return mx.sum(result, axis=1)


def _input_rows(x, indices, features):
    if x.shape[-1] != features:
        raise ValueError("activation width does not match in_features")
    if x.shape[:-1] == indices.shape:
        return True
    if x.shape[:-1] == indices.shape[:-1]:
        return False
    raise ValueError("activation rows do not match the expert indices")


def _prep(x, ids, in_gate, in_up, features, slots, per_slot, projections=1):
    import mlx.core as mx

    return _transform_kernel("prep", per_slot)(
        inputs=[x.reshape(-1, features), ids, in_gate, in_up, _dims(features, 0, ids.size, 0, slots)],
        grid=(features // 128 * 32, ids.size, projections),
        threadgroup=(128, 1, 1),
        output_shapes=[(projections * ids.size, features)],
        output_dtypes=[mx.float32],
    )[0]


def _finish(inner, ids, out_scale, features, splits, dtype):
    return _transform_kernel("finish")(
        inputs=[inner, ids, out_scale, _dims(features, 0, ids.size, splits)],
        template=[("T", dtype)],
        grid=(features // 128 * 32, ids.size, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(ids.size, features)],
        output_dtypes=[dtype],
    )[0]


def _finish_fused(inner, ids, out_scale, features, scores, residual):
    return _transform_kernel("finish_fused")(
        inputs=[inner, ids, out_scale, scores, residual, _dims(features)],
        template=[("T", residual.dtype)],
        grid=(features // 128 * 288, ids.size // 9, 1),
        threadgroup=(288, 1, 1),
        output_shapes=[residual.shape], output_dtypes=[residual.dtype],
    )[0]


def fast_expert_linear(x, indices, packed, in_scale, out_scale, k, in_features, out_features, sharded_side=1):
    """Apply the exact EXLR8 matrix to each selected expert row."""
    import mlx.core as mx

    experts = _check_format(packed, k, in_features, out_features, sharded_side)
    if in_scale.shape != (experts, in_features) or out_scale.shape != (experts, out_features):
        raise ValueError("expert side vectors must contain full signed channel scales")
    per_slot = _input_rows(x, indices, in_features)
    ids = indices.reshape(-1).astype(mx.uint32)
    xh = _prep(x, ids, in_scale, in_scale, in_features, indices.shape[-1], per_slot)
    splits = min(8, max(1, in_features // 768)) if sharded_side != 2 else 1
    inner = _partials(xh, ids, packed, k, in_features, out_features, sharded_side, splits)
    return _finish(inner, ids, out_scale, out_features, splits, x.dtype).reshape(indices.shape + (out_features,))


def _expert_mlp(x, indices, packed, scales, k, hidden, intermediate, config, lookup=False, reuse=1, fp8=None, expert_to_slot=None, direct_x=False, scores=None, residual=None):
    """Apply selected gate/up/down experts in five GPU dispatches.

    The signed side vectors and 128-channel transforms use float32. Gate,
    up, SwiGLU, and final outputs retain the activation type used by the
    existing expert path. ``packed`` and ``scales`` use the same projection
    dictionaries as ``ExpertLayer``. ``scores`` and ``residual`` select the
    one-split finish with eight routed experts and one shared expert.
    """
    import mlx.core as mx

    per_slot = _input_rows(x, indices, hidden)
    for projection in ("gate", "up", "down"):
        a, b, side = (intermediate, hidden, 2) if projection == "down" else (hidden, intermediate, 1)
        experts = _check_format(packed[projection], k, a, b, side)
        if scales[projection][0].shape != (experts, a) or scales[projection][1].shape != (experts, b):
            raise ValueError("expert side vectors must contain full signed channel scales")
    ids = indices.reshape(-1).astype(mx.uint32)
    splits, nsg, down_splits, down_nsg, quad, direct, vector = config
    xh = _prep(x, ids, scales["gate"][0], scales["up"][0], hidden, indices.shape[-1], per_slot, 2)
    inner = _partials(xh, ids, packed["gate"], k, hidden, intermediate, 1, splits, nsg, quad,
                      packed_up=packed["up"], direct=direct, vector=vector, lookup=lookup, reuse=reuse,
                      fp8=fp8["gate"] if fp8 is not None else None,
                      fp8_up=fp8["up"] if fp8 is not None else None, expert_to_slot=expert_to_slot, direct_x=direct_x)
    down_xh = _transform_kernel("gate_finish")(
        inputs=[inner, ids, scales["gate"][1], scales["up"][1], scales["down"][0], _dims(intermediate, 0, ids.size, splits)],
        template=[("T", x.dtype)],
        grid=(intermediate // 128 * 32, ids.size, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(ids.size, intermediate)],
        output_dtypes=[mx.float32],
    )[0]
    down_inner = _partials(down_xh, ids, packed["down"], k, intermediate, hidden, 2, down_splits, down_nsg,
                           quad, direct=direct, vector=vector, lookup=lookup, reuse=reuse,
                           fp8=fp8["down"] if fp8 is not None else None, expert_to_slot=expert_to_slot, direct_x=direct_x)
    if scores is not None:
        return _finish_fused(down_inner, ids, scales["down"][1], hidden, scores, residual)
    return _finish(down_inner, ids, scales["down"][1], hidden, down_splits, x.dtype).reshape(indices.shape + (hidden,))


def fast_expert_mlp(x, indices, packed, scales, k, hidden, intermediate, *, reuse=False, fp8=None, expert_to_slot=None):
    """Apply selected EXLR8 expert MLPs; return [*indices.shape, hidden].

    ``reuse=True`` shares decoded tiles across 2–4 token rows. Other row
    counts use the normal path. All dot products keep float32 arithmetic.
    ``fp8`` supplies cached gate/up/down bytes in trellis tile order.
    ``expert_to_slot`` maps source IDs to slots, with 0xFFFFFFFF on a miss.
    """
    splits = min(8, max(1, hidden // 768))
    rows = indices.size // indices.shape[-1]
    reuse_rows = rows if reuse and 2 <= rows <= 4 else 1
    return _expert_mlp(x, indices, packed, scales, k, hidden, intermediate,
                       (splits, 4, 1, 4, True, True, True), reuse=reuse_rows,
                       fp8=fp8, expert_to_slot=expert_to_slot, direct_x=(k == 4))


def compiled_expert_mlp_finish(expert):
    """Compile one to four K4 rows with eight routed experts and one shared expert.

    Return the typed routed sum, shared sum, and residual sum. RMS, routing,
    attention, and cache updates stay outside the compiled call.
    """
    import mlx.core as mx

    h, m = expert.hidden, expert.intermediate
    splits = min(8, max(1, h // 768))

    def apply(x, indices, scores, residual):
        shape = residual.shape
        rows = x.size // h
        return _expert_mlp(x.reshape(rows, h), indices.reshape(rows, 9), expert.packed, expert.scales,
                           4, h, m, (splits, 4, 1, 4, True, True, True), direct_x=True,
                           scores=scores.reshape(rows, 8), residual=residual.reshape(rows, h)).reshape(shape)

    return mx.compile(apply)
