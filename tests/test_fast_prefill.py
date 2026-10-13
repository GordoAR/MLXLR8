"""Grouped prefill checks for decode geometry, row tails, and MLP order."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from exlr8.trellis import decode_trellis_reference, dequantize_reference, hadamard_reference


metal = pytest.mark.skipif(
    os.environ.get("EXLR8_TEST_METAL") != "1",
    reason="set EXLR8_TEST_METAL=1 on a Metal host",
)


def test_e4m3_operands_are_exact_in_float16():
    codes = np.arange(127, dtype=np.int32)
    exponent, fraction = codes >> 3, codes & 7
    levels = np.where(
        exponent == 0,
        fraction * 2.0**-9,
        (8 + fraction) * np.exp2(exponent - 10),
    ).astype(np.float32)
    values = np.concatenate([levels, -levels])
    np.testing.assert_array_equal(values.astype(np.float16).astype(np.float32), values)


@metal
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("per_slot", [False, True])
def test_grouped_linear_matches_independent_cpu_decode(k, side, per_slot):
    import mlx.core as mx
    from exlr8.fast_prefill import grouped_expert_linear

    rng = np.random.default_rng(39)
    experts, width, output = 4, 256, 384
    size = width * output * k // 8
    packed = rng.integers(0, 256, (experts, size), dtype=np.uint8)
    in_scale = rng.uniform(-0.1, 0.1, (experts, width)).astype(np.float16)
    out_scale = rng.uniform(-0.1, 0.1, (experts, output)).astype(np.float16)
    ids = np.array([[3, 0, 3], [1, 0, 2], [0, 3, 1]], dtype=np.uint32)
    x = rng.normal(size=(3, 3, width) if per_slot else (3, width)).astype(np.float32)
    matrices = np.stack([
        dequantize_reference(packed[e], in_scale[e], out_scale[e], k, width, output, side)
        for e in range(experts)
    ])
    rows = x if per_slot else np.broadcast_to(x[:, None, :], (3, 3, width))
    expected = np.einsum("tsi,tsoi->tso", rows, matrices[ids])
    actual = grouped_expert_linear(
        mx.array(x), mx.array(ids), mx.array(packed), mx.array(in_scale), mx.array(out_scale),
        k, width, output, side,
    )
    np.testing.assert_allclose(np.array(actual), expected, rtol=5e-5, atol=3e-6)


@metal
@pytest.mark.parametrize("tokens", [3, 12, 40, 129])
def test_row_blocks_and_ragged_segments(tokens):
    import mlx.core as mx
    from exlr8.fast_prefill import grouped_expert_linear
    from exlr8.trellis import expert_linear

    rng = np.random.default_rng(41)
    packed = mx.array(rng.integers(0, 256, (3, 8192), dtype=np.uint8))
    in_scale = mx.array(rng.uniform(-0.1, 0.1, (3, 128)).astype(np.float16))
    out_scale = mx.array(rng.uniform(-0.1, 0.1, (3, 128)).astype(np.float16))
    x = mx.array(rng.normal(size=(tokens, 128)).astype(np.float32))
    ids_np = np.zeros((tokens, 2), dtype=np.uint32)
    ids_np[::3, 0] = 2
    # Expert 1 is absent. Expert 0 has a segment longer than one row block.
    ids = mx.array(ids_np)
    actual = grouped_expert_linear(x, ids, packed, in_scale, out_scale, 4, 128, 128)
    expected = expert_linear(x, ids, packed, in_scale, out_scale, 4, 128, 128)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=5e-5, atol=2e-6)


@metal
@pytest.mark.parametrize("side,width,output", [(1, 6144, 2048), (2, 2048, 6144)])
@pytest.mark.parametrize("k", [3, 4])
def test_model_projection_dimensions(side, width, output, k):
    import mlx.core as mx
    from exlr8.fast_prefill import grouped_expert_linear
    from exlr8.trellis import expert_linear

    rng = np.random.default_rng(47)
    packed = mx.array(rng.integers(0, 256, (3, width * output * k // 8), dtype=np.uint8))
    in_scale = mx.array(rng.uniform(-0.03, 0.03, (3, width)).astype(np.float16))
    out_scale = mx.array(rng.uniform(-0.03, 0.03, (3, output)).astype(np.float16))
    x = mx.array(rng.normal(size=(4, width)).astype(np.float32))
    ids = mx.array([[2, 0, 2], [0, 2, 0], [2, 0, 0], [0, 2, 2]], dtype=mx.uint32)
    actual = grouped_expert_linear(x, ids, packed, in_scale, out_scale, k, width, output, side)
    expected = expert_linear(x, ids, packed, in_scale, out_scale, k, width, output, side)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=5e-4, atol=3e-6)


@metal
@pytest.mark.parametrize("tokens", [2, 65])
@pytest.mark.parametrize("side", [1, 2])
def test_dense_prefill_without_expert_sort(tokens, side):
    import mlx.core as mx
    from exlr8.fast_prefill import trellis_linear_prefill
    from exlr8.trellis import trellis_linear

    rng = np.random.default_rng(51)
    packed = mx.array(rng.integers(0, 256, 128 * 256 // 2, dtype=np.uint8))
    suh = mx.array(rng.uniform(-0.1, 0.1, 128).astype(np.float16))
    svh = mx.array(rng.uniform(-0.1, 0.1, 256).astype(np.float16))
    x = mx.array(rng.normal(size=(1, tokens, 128)).astype(np.float32))
    actual = trellis_linear_prefill(x, packed, suh, svh, 4, 128, 256, side)
    expected = trellis_linear(x, packed, suh, svh, 4, 128, 256, side)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=5e-5, atol=2e-6)


@metal
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("half_dot", [False, True])
def test_grouped_mlp_preserves_activation_dtype_and_expert_order(dtype, half_dot):
    import mlx.core as mx
    from mlx_lm.models.activations import swiglu
    from exlr8.fast_prefill import grouped_expert_linear, grouped_expert_mlp
    from exlr8.trellis import expert_linear

    rng = np.random.default_rng(53)
    width, intermediate, experts = 256, 128, 5
    packed = {}
    scales = {}
    for projection in ("gate", "up", "down"):
        source, target = (intermediate, width) if projection == "down" else (width, intermediate)
        packed[projection] = mx.array(rng.integers(0, 256, (experts, source * target // 2), dtype=np.uint8))
        scales[projection] = (
            mx.array(rng.uniform(-0.1, 0.1, (experts, source)).astype(np.float16)),
            mx.array(rng.uniform(-0.1, 0.1, (experts, target)).astype(np.float16)),
        )
    ids = mx.array([[4, 0, 2], [3, 4, 0], [0, 1, 4], [4, 4, 2]], dtype=mx.uint32)
    x = mx.array(rng.normal(size=(4, width)).astype(np.float32)).astype(getattr(mx, dtype))
    linear = grouped_expert_linear if half_dot else expert_linear
    options = {"half_dot": True} if half_dot else {}
    gu = [linear(x, ids, packed[p], *scales[p], 4, width, intermediate, 1, **options) for p in ("gate", "up")]
    expected = linear(swiglu(*gu), ids, packed["down"], *scales["down"], 4, intermediate, width, 2, **options)
    actual = grouped_expert_mlp(x, ids, packed, scales, 4, width, intermediate, half_dot=half_dot)
    assert actual.dtype == x.dtype
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)),
        rtol=0.01 if dtype == "bfloat16" else 3e-4, atol=2e-7,
    )


@metal
@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("tokens", [3, 40])
def test_half_dot_matches_independent_cpu_operands(k, side, tokens):
    import mlx.core as mx
    from exlr8.fast_prefill import grouped_expert_linear

    rng = np.random.default_rng(61)
    experts, width, output = 3, 256, 384
    packed = rng.integers(0, 256, (experts, width * output * k // 8), dtype=np.uint8)
    in_scale = rng.uniform(-0.1, 0.1, (experts, width)).astype(np.float16)
    out_scale = rng.uniform(-0.1, 0.1, (experts, output)).astype(np.float16)
    base_ids = np.array([[2, 0, 2], [0, 1, 0], [2, 0, 1]], dtype=np.uint32)
    ids = np.tile(base_ids, ((tokens + 2) // 3, 1))[:tokens]
    x = rng.normal(size=(tokens, 3, width)).astype(np.float32)
    matrices = np.stack([
        decode_trellis_reference(packed[e], k, width, output, side).astype(np.float32)
        for e in range(experts)
    ])
    xh = hadamard_reference(x * in_scale[ids].astype(np.float32)).astype(np.float16).astype(np.float32)
    expected = hadamard_reference(np.einsum("tsi,tsio->tso", xh, matrices[ids])) * out_scale[ids].astype(np.float32)
    actual = np.array(grouped_expert_linear(
        mx.array(x), mx.array(ids), mx.array(packed), mx.array(in_scale), mx.array(out_scale),
        k, width, output, side, half_dot=True,
    ))
    # CPU and SIMD Hadamard addition order can differ at one FP16 tie.
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-5
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-6)


@metal
@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("side,width,output", [(1, 6144, 2048), (2, 2048, 6144)])
def test_half_dot_model_dimensions_match_cpu_operands(k, side, width, output):
    import mlx.core as mx
    from exlr8.fast_prefill import grouped_expert_linear

    rng = np.random.default_rng(67)
    packed = rng.integers(0, 256, (1, width * output * k // 8), dtype=np.uint8)
    in_scale = rng.uniform(-0.03, 0.03, (1, width)).astype(np.float16)
    out_scale = rng.uniform(-0.03, 0.03, (1, output)).astype(np.float16)
    x = rng.normal(size=(2, width)).astype(np.float32)
    matrix = decode_trellis_reference(packed[0], k, width, output, side).astype(np.float32)
    xh = hadamard_reference(x * in_scale.astype(np.float32)).astype(np.float16).astype(np.float32)
    expected = (hadamard_reference(xh @ matrix) * out_scale.astype(np.float32))[:, None, :]
    actual = np.array(grouped_expert_linear(
        mx.array(x), mx.zeros((2, 1), dtype=mx.uint32), mx.array(packed),
        mx.array(in_scale), mx.array(out_scale), k, width, output, side, half_dot=True,
    ))
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-5
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=5e-6)


@metal
def test_half_dot_keeps_float32_accumulation():
    import mlx.core as mx
    from exlr8.fast_prefill import grouped_expert_linear
    from exlr8.trellis import mcg_codebook

    width, output = 6144, 128
    x = np.zeros((2, width), dtype=np.float32)
    x[:, ::128] = np.float32(10 * np.sqrt(128))
    packed = mx.zeros((1, width * output // 2), dtype=mx.uint8)
    in_scale = mx.ones((1, width), dtype=mx.float32)
    out_scale = mx.ones((1, output), dtype=mx.float32)
    actual = np.array(grouped_expert_linear(
        mx.array(x), mx.zeros((2, 1), dtype=mx.uint32), packed, in_scale, out_scale,
        4, width, output, half_dot=True,
    ))
    inner_sum = width * 10 * float(mcg_codebook()[0])
    assert inner_sum > float(np.finfo(np.float16).max)
    expected = np.zeros((2, 1, output), dtype=np.float32)
    expected[:, :, 0] = inner_sum * np.sqrt(128)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.01)


@metal
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("block_rows", [8, 16, 32])
def test_coalesced_k4_preserves_exact_mlp_and_chain_bits(monkeypatch, dtype, block_rows):
    import mlx.core as mx
    import exlr8.fast_prefill as prefill

    directory = Path(prefill.__file__).with_name("metal")
    source = (directory / "fast_prefill.metal").read_text()
    for guard, forced in (("#if K_BITS != 4 || BLOCK_ROWS > 32", "#if 1"),
                          ("#if K_BITS == 4 && BLOCK_ROWS <= 32", "#if 0")):
        assert source.count(guard) == 1
        source = source.replace(guard, forced)

    @lru_cache(maxsize=2)
    def generic_kernel(k, side, block_rows, gate_up=False, half_dot=False):
        assert k == 4 and block_rows in (8, 16, 32) and not half_dot
        names = (["xh", "packed_gate", "packed_up", "table", "count", "dims"] if gate_up
                 else ["xh", "packed", "table", "count", "dims"])
        prefix = (f"#define K_BITS {k}u\n#define OUTPUT_GROUPED {int(side != 2)}\n"
                  f"#define BLOCK_ROWS {block_rows}u\n#define GATE_UP {int(gate_up)}\n#define HALF_DOT 0\n")
        return mx.fast.metal_kernel(name=f"exlr8_test_prefill_generic_s{side}_m{block_rows}_gu{int(gate_up)}",
            input_names=names, output_names=["out"], source=prefix + source,
            header="#include <metal_simdgroup_matrix>\n" + (directory / "fast_decode_common.metal").read_text())

    rng = np.random.default_rng(659)
    width, intermediate, experts = 256, 128, 5
    packed, scales = {}, {}
    for projection in ("gate", "up", "down"):
        a, b = (intermediate, width) if projection == "down" else (width, intermediate)
        packed[projection] = mx.array(rng.integers(0, 256, (experts, a * b // 2), np.uint8))
        scales[projection] = (
            mx.array(rng.uniform(-0.1, 0.1, (experts, a)).astype(np.float16)),
            mx.array(rng.uniform(-0.1, 0.1, (experts, b)).astype(np.float16)),
        )
    ids = mx.array(np.tile([[4, 0, 4], [0, 1, 0], [4, 3, 1]], (22, 1))[:65], mx.uint32)
    x = mx.array(rng.normal(size=(65, width)).astype(np.float32)).astype(getattr(mx, dtype))
    monkeypatch.setattr(prefill, "_block_rows", lambda rows, experts: block_rows)

    def apply(value):
        return prefill.grouped_expert_mlp(value, ids, packed, scales, 4, width, intermediate)

    def outputs():
        single, state = apply(x), x
        for _ in range(3):
            state = state + mx.mean(apply(state), axis=-2) * 0.001
        mx.eval(single, state)
        return single, state

    actual = outputs()
    with monkeypatch.context() as context:
        context.setattr(prefill, "_kernel", generic_kernel)
        expected = outputs()
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(np.array(a.astype(mx.float32)).view(np.uint32),
                                      np.array(e.astype(mx.float32)).view(np.uint32))
