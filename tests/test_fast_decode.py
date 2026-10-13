"""Exact codebook and selected expert tests for the fused decode path."""

import os
import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

from exlr8.trellis import mcg_codebook


def _fp8_half_bits():
    source = Path(__file__).resolve().parents[1] / "exlr8/metal/fast_decode_fp8.metal"
    return np.array([int(v,16) for v in re.findall(r"0x([0-9a-f]{4})u",source.read_text())],dtype=np.uint16)


def _mcg_fp8_codes():
    values = mcg_codebook()
    bits = values.view(np.uint16)
    magnitude = bits&32767
    normal = (magnitude.astype(np.uint32)-8192)>>7
    subnormal = np.rint(np.abs(values.astype(np.float32))*512).astype(np.uint16)
    return (((bits>>8)&128)|np.where(magnitude<0x2400,subnormal,normal)).astype(np.uint8)


def test_fp8_byte_table_keeps_e4m3_values():
    table = _fp8_half_bits()
    assert table.size == 256
    code = np.arange(256,dtype=np.uint16)
    exponent, fraction = (code>>3)&15, code&7
    magnitude = np.where(exponent==0,fraction/512,np.ldexp(1+fraction/8,exponent.astype(np.int32)-7))
    magnitude[(code&127)==127] = np.nan
    expected = np.copysign(magnitude,np.where(code&128,-1,1)).astype(np.float16)
    np.testing.assert_array_equal(table,expected.view(np.uint16))


def test_fp8_byte_table_keeps_all_mcg_codewords():
    np.testing.assert_array_equal(_fp8_half_bits()[_mcg_fp8_codes()],mcg_codebook().view(np.uint16))


def test_half_bit_round_matches_all_mcg_codewords():
    states = np.arange(65536, dtype=np.uint32)
    product = ((states * np.uint32(0xCBAC1FED)) & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    summed = (product & 65535).astype(np.uint16).view(np.float16)
    summed = summed + (product >> 16).astype(np.uint16).view(np.float16)
    bits = summed.view(np.uint16).astype(np.uint32)
    magnitude = bits & 32767
    rounded = (bits + 63 + ((bits >> 7) & 1)) & 65408
    normal = rounded.astype(np.uint16).view(np.float16)
    values = np.where(magnitude < 0x2400, np.rint(summed.astype(np.float32) * 512) / 512, normal)
    np.testing.assert_array_equal(values.astype(np.float16).view(np.uint16), mcg_codebook().view(np.uint16))


def test_direct_k4_words_match_reference_bitstream():
    source = Path(__file__).resolve().parents[1] / "third_party/ponyexl3/ref/trellis.py"
    spec = importlib.util.spec_from_file_location("fast_decode_trellis_reference",source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    canonical = np.random.default_rng(235).integers(0,65536,(16,16,64),dtype=np.uint16)
    words = canonical.view(np.uint32)
    prior = np.roll(words,1,axis=-1)
    values = np.stack([((prior<<4)|(words>>28))&65535,
                       ((prior<<8)|(words>>24))&65535,
                       ((prior<<12)|(words>>20))&65535,
                       words>>16,(words>>12)&65535,(words>>8)&65535,
                       (words>>4)&65535,words&65535],axis=-1)
    np.testing.assert_array_equal(values.reshape(16,16,256),reference.unpack_trellis(canonical,4))


metal = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1", reason="set EXLR8_TEST_METAL=1 on a Metal host")


@metal
def test_metal_fp8_bytes_keep_finite_values_and_signed_zero():
    import mlx.core as mx

    source = Path(__file__).resolve().parents[1] / "exlr8/metal/fast_decode_fp8.metal"
    kernel = mx.fast.metal_kernel(
        name="exlr8_test_fp8_bytes",input_names=["codes"],output_names=["out"],
        source="uint i=thread_position_in_grid.x; out[i]=exlr8_fp8_byte(codes[i]);",
        header=source.read_text(),
    )
    actual = kernel(inputs=[mx.arange(256,dtype=mx.uint32)],grid=(256,1,1),threadgroup=(256,1,1),
                    output_shapes=[(256,)],output_dtypes=[mx.float16])[0]
    actual = np.array(actual)
    finite = (np.arange(256)&127)!=127
    np.testing.assert_array_equal(actual.view(np.uint16)[finite],_fp8_half_bits()[finite])
    assert np.isnan(actual[~finite]).all()


@metal
@pytest.mark.parametrize("vector", [False, True])
def test_metal_codebook_matches_all_mcg_codewords(vector):
    import mlx.core as mx

    source = Path(__file__).resolve().parents[1] / "exlr8/metal/fast_decode_common.metal"
    body = ("uint i = thread_position_in_grid.x; float2 value = exlr8_decode2(uint2(states[i*2u], states[i*2u+1u]));"
            "out[i*2u] = value.x; out[i*2u+1u] = value.y;" if vector else
            "uint i = thread_position_in_grid.x; out[i] = exlr8_decode(states[i]);")
    kernel = mx.fast.metal_kernel(
        name=f"exlr8_test_fast_codebook_v{int(vector)}",
        input_names=["states"],
        output_names=["out"],
        source=body,
        header=source.read_text(),
    )
    actual = kernel(
        inputs=[mx.arange(65536, dtype=mx.uint32)],
        grid=(32768 if vector else 65536, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(65536,)],
        output_dtypes=[mx.float32],
    )[0]
    np.testing.assert_array_equal(np.array(actual).view(np.uint32), mcg_codebook().astype(np.float32).view(np.uint32))


def _fixture(mx, hidden=256, intermediate=384, experts=3, k=4):
    rng = np.random.default_rng(181)
    packed, scales = {}, {}
    for projection in ("gate", "up", "down"):
        a, b = (intermediate, hidden) if projection == "down" else (hidden, intermediate)
        packed[projection] = mx.array(rng.integers(0, 256, experts * a * b * k // 8, dtype=np.uint8))
        scales[projection] = (
            mx.array(rng.uniform(-0.08, 0.08, (experts, a)).astype(np.float16)),
            mx.array(rng.uniform(-0.5, 0.5, (experts, b)).astype(np.float16)),
        )
    return packed, scales


def _fp8_fixture(mx, packed, k, hidden, intermediate, cached):
    matrix_bytes = hidden*intermediate*k//8
    experts = packed["gate"].nbytes//matrix_bytes
    mapping = np.full(experts,0xFFFFFFFF,dtype=np.uint32)
    mapping[cached] = np.arange(len(cached),dtype=np.uint32)
    values = {}
    index = np.arange(256,dtype=np.int32)
    end = (index+1)*k+256*k
    last = (end-1)//32
    codebook = _mcg_fp8_codes()
    for projection in ("gate","up","down"):
        words = np.array(packed[projection]).view(np.uint32).reshape(experts,-1,8*k)
        words = words[cached]
        merged = words[..., (last-1)%(8*k)].astype(np.uint64)<<np.uint64(32)
        merged |= words[...,last%(8*k)].astype(np.uint64)
        codewords = ((merged>>((last+1)*32-end).astype(np.uint64))&65535).astype(np.uint16)
        values[projection] = mx.array(codebook[codewords].reshape(-1))
    return values,mx.array(mapping)


def _reference(x, ids, packed, scales, k, hidden, intermediate):
    from mlx_lm.models.activations import swiglu
    from exlr8.trellis import expert_linear

    gate = expert_linear(x, ids, packed["gate"], *scales["gate"], k, hidden, intermediate, 1)
    up = expert_linear(x, ids, packed["up"], *scales["up"], k, hidden, intermediate, 1)
    return expert_linear(swiglu(gate, up), ids, packed["down"], *scales["down"], k, intermediate, hidden, 2)


@metal
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("per_slot", [False, True])
@pytest.mark.parametrize("lookup", [False, True])
def test_fused_expert_mlp_matches_reference(k, per_slot, lookup):
    import mlx.core as mx
    from exlr8.fast_decode import _expert_mlp

    packed, scales = _fixture(mx, k=k)
    ids = mx.array([[2, 0], [1, 2]], dtype=mx.uint32)
    shape = (2, 2, 256) if per_slot else (2, 256)
    x = mx.array(np.random.default_rng(29).normal(size=shape).astype(np.float32))
    expected = _reference(x, ids, packed, scales, k, 256, 384)
    actual = _expert_mlp(x, ids, packed, scales, k, 256, 384, (1,4,1,4,True,True,True),lookup)
    assert actual.shape == (2, 2, 256)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=7e-5, atol=5e-7)


@metal
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("lookup", [False, True])
def test_full_glm_expert_dimensions(dtype_name, lookup):
    import mlx.core as mx
    from exlr8.fast_decode import _expert_mlp, fast_expert_mlp

    packed, scales = _fixture(mx, hidden=6144, intermediate=2048, experts=9)
    ids = mx.arange(9, dtype=mx.uint32).reshape(1, 9)
    dtype = getattr(mx, dtype_name)
    x = mx.array(np.random.default_rng(32).normal(size=(1, 6144)), dtype=dtype)
    expected = _reference(x, ids, packed, scales, 4, 6144, 2048).astype(mx.float32)
    inline = fast_expert_mlp(x, ids, packed, scales, 4, 6144, 2048).astype(mx.float32)
    actual = (_expert_mlp(x, ids, packed, scales, 4, 6144, 2048, (8,4,1,4,True,True,True),True).astype(mx.float32)
              if lookup else inline)
    if lookup:
        np.testing.assert_array_equal(np.array(actual), np.array(inline))
    diff = np.array(actual) - np.array(expected)
    relative_l2 = np.linalg.norm(diff) / np.linalg.norm(np.array(expected))
    limit = 3e-6 if dtype_name == "float32" else 0.001 if dtype_name == "float16" else 0.007
    assert relative_l2 < limit


@metal
@pytest.mark.parametrize("side", [1, 2])
def test_fused_linear_matches_reference(side):
    import mlx.core as mx
    from exlr8.fast_decode import fast_expert_linear
    from exlr8.trellis import expert_linear

    packed, scales = _fixture(mx)
    projection, a, b = ("down", 384, 256) if side == 2 else ("gate", 256, 384)
    ids = mx.array([[2, 0], [1, 2]], dtype=mx.uint32)
    x = mx.array(np.random.default_rng(77).normal(size=(2, a)), dtype=mx.float32)
    expected = expert_linear(x, ids, packed[projection], *scales[projection], 4, a, b, side)
    actual = fast_expert_linear(x, ids, packed[projection], *scales[projection], 4, a, b, side)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=5e-5, atol=1e-6)


@metal
@pytest.mark.parametrize("rows", [2, 3, 4])
@pytest.mark.parametrize("pattern", ["shared", "all", "changing", "duplicates"])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_multirow_reuse_keeps_each_output(rows, pattern, dtype_name):
    import mlx.core as mx
    from exlr8.fast_decode import fast_expert_mlp

    packed, scales = _fixture(mx, experts=40)
    if pattern == "shared":
        routed = np.arange(rows*8).reshape(rows,8)
        ids = np.concatenate([routed,np.full((rows,1),39)],axis=-1)
    elif pattern == "all":
        ids = np.stack([np.roll(np.arange(9),r) for r in range(rows)])
    elif pattern == "changing":
        routed = np.stack([np.arange(r*4,r*4+8) for r in range(rows)])
        ids = np.concatenate([routed,np.full((rows,1),39)],axis=-1)
    else:
        ids = np.tile([0,0,1,1,2,2,0,1,2],(rows,1))
    ids = mx.array(ids,dtype=mx.uint32)
    x = mx.array(np.random.default_rng(431).normal(size=(rows,256)),dtype=getattr(mx,dtype_name))
    expected = fast_expert_mlp(x,ids,packed,scales,4,256,384)
    actual = fast_expert_mlp(x,ids,packed,scales,4,256,384,reuse=True)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))


@metal
def test_multirow_reuse_k3_per_slot_outputs():
    import mlx.core as mx
    from exlr8.fast_decode import fast_expert_mlp

    packed, scales = _fixture(mx,k=3)
    ids = mx.array([[0,1,2],[2,0,1],[1,2,0]],dtype=mx.uint32)
    x = mx.array(np.random.default_rng(432).normal(size=(3,3,256)),dtype=mx.bfloat16)
    expected = fast_expert_mlp(x,ids,packed,scales,3,256,384)
    actual = fast_expert_mlp(x,ids,packed,scales,3,256,384,reuse=True)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))


@metal
@pytest.mark.parametrize("rows", [2, 4])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_multirow_reuse_full_glm_dimensions(rows, dtype_name):
    import mlx.core as mx
    from exlr8.fast_decode import fast_expert_mlp

    packed, scales = _fixture(mx,hidden=6144,intermediate=2048,experts=9)
    ids = mx.array(np.stack([np.roll(np.arange(9),r) for r in range(rows)]),dtype=mx.uint32)
    x = mx.array(np.random.default_rng(433).normal(size=(rows,6144)),dtype=getattr(mx,dtype_name))
    expected = fast_expert_mlp(x,ids,packed,scales,4,6144,2048)
    actual = fast_expert_mlp(x,ids,packed,scales,4,6144,2048,reuse=True)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))


@metal
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_multirow_reuse_pairs_full_glm_dimensions(dtype_name):
    import mlx.core as mx
    from exlr8.fast_decode import _expert_mlp, fast_expert_mlp

    packed, scales = _fixture(mx,hidden=6144,intermediate=2048,experts=9)
    ids = mx.array(np.stack([np.roll(np.arange(9),r) for r in range(4)]),dtype=mx.uint32)
    x = mx.array(np.random.default_rng(434).normal(size=(4,6144)),dtype=getattr(mx,dtype_name))
    expected = fast_expert_mlp(x,ids,packed,scales,4,6144,2048)
    actual = _expert_mlp(x,ids,packed,scales,4,6144,2048,(8,4,1,4,True,True,True),reuse=2)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))


@metal
@pytest.mark.parametrize("k", [2,3,4])
@pytest.mark.parametrize("rows", [1,4])
@pytest.mark.parametrize("mode", ["miss","mixed","hit"])
@pytest.mark.parametrize("dtype_name", ["float32","bfloat16"])
def test_static_fp8_cache_keeps_current_outputs(k, rows, mode, dtype_name):
    import mlx.core as mx
    from exlr8.fast_decode import fast_expert_mlp

    packed,scales = _fixture(mx,k=k)
    cached = [2] if mode=="miss" else [2,0] if mode=="mixed" else [2,0,1]
    fp8,mapping = _fp8_fixture(mx,packed,k,256,384,cached)
    if mode=="miss":
        mapping = mx.full((3,),0xFFFFFFFF,dtype=mx.uint32)
    ids = mx.array([[2,0,1],[0,2,1],[1,2,0],[2,1,0]][:rows],dtype=mx.uint32)
    x = mx.array(np.random.default_rng(534).normal(size=(rows,256)),dtype=getattr(mx,dtype_name))
    expected = fast_expert_mlp(x,ids,packed,scales,k,256,384)
    actual = fast_expert_mlp(x,ids,packed,scales,k,256,384,fp8=fp8,expert_to_slot=mapping)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))


@metal
def test_static_fp8_cache_with_four_row_reuse():
    import mlx.core as mx
    from exlr8.fast_decode import fast_expert_mlp

    packed,scales = _fixture(mx)
    fp8,mapping = _fp8_fixture(mx,packed,4,256,384,[2,0])
    ids = mx.array([[2,0,1],[0,2,1],[1,2,0],[2,1,0]],dtype=mx.uint32)
    x = mx.array(np.random.default_rng(535).normal(size=(4,256)),dtype=mx.bfloat16)
    expected = fast_expert_mlp(x,ids,packed,scales,4,256,384,reuse=True)
    actual = fast_expert_mlp(x,ids,packed,scales,4,256,384,reuse=True,fp8=fp8,expert_to_slot=mapping)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))


@metal
@pytest.mark.parametrize("dtype_name", ["float32","bfloat16"])
def test_static_fp8_cache_full_glm_dimensions(dtype_name):
    import mlx.core as mx
    from exlr8.fast_decode import fast_expert_mlp

    packed,scales = _fixture(mx,hidden=6144,intermediate=2048,experts=9)
    fp8,mapping = _fp8_fixture(mx,packed,4,6144,2048,[8])
    ids = mx.arange(9,dtype=mx.uint32).reshape(1,9)
    x = mx.array(np.random.default_rng(536).normal(size=(1,6144)),dtype=getattr(mx,dtype_name))
    expected = fast_expert_mlp(x,ids,packed,scales,4,6144,2048)
    actual = fast_expert_mlp(x,ids,packed,scales,4,6144,2048,fp8=fp8,expert_to_slot=mapping)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.array(expected.astype(mx.float32)))


def test_direct_activation_pairs_match_lane_shuffles():
    lane = np.arange(32, dtype=np.uint32)
    values = np.arange(16, dtype=np.float32) + np.float32(0.25)
    lane_values = values[lane & 15]
    base = (lane & 3) * 2
    assert np.all(base % 2 == 0)
    for j in range(4):
        index = (lane & 3) * 2 + (j & 1) + (j & 2) * 4
        expected = lane_values[index]
        actual = values[base + (0, 1, 8, 9)[j]]
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("k", [3, 4])
def test_mlp_uses_direct_activations_only_for_k4(monkeypatch, k):
    import exlr8.fast_decode as decode

    seen = {}
    output = object()

    def apply(*args, **kwargs):
        seen.update(kwargs)
        return output

    monkeypatch.setattr(decode, "_expert_mlp", apply)
    ids = np.arange(9, dtype=np.uint32).reshape(1, 9)
    assert decode.fast_expert_mlp(None, ids, None, None, k, 6144, 2048) is output
    assert seen["direct_x"] == (k == 4)


@metal
@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("change, expected", [
    ({}, True), ({"k": 3}, False), ({"in_features": 128}, False),
    ({"out_features": 128}, False), ({"rows": 18}, False),
    ({"splits": 2}, False), ({"nsg": 8}, False), ({"quad": False}, False),
    ({"direct": False}, False), ({"vector": False}, False),
    ({"lookup": True}, False), ({"reuse": 2}, False),
    ({"direct_x": False}, False), ({"hot": True}, False),
    ({"swap_projection": True}, False),
])
def test_fixed_dimension_dispatch_bounds(monkeypatch, side, change, expected):
    import mlx.core as mx
    import exlr8.fast_decode as decode

    options = dict(k=4, in_features=6144 if side == 1 else 2048,
                   out_features=2048 if side == 1 else 6144, side=side,
                   splits=8 if side == 1 else 1, direct_x=True)
    options.update(change)
    rows = options.pop("rows", 9)
    packed = mx.zeros((4,), dtype=mx.uint8)
    up = (side == 1) != options.pop("swap_projection", False)
    options["packed_up"] = packed if up else None
    if options.pop("hot", False):
        options.update(fp8=packed, fp8_up=packed, expert_to_slot=mx.zeros((9,), dtype=mx.uint32))
    xh = mx.zeros(((2 if up else 1) * rows, options["in_features"]))
    ids = mx.arange(rows, dtype=mx.uint32)
    seen, output = [], object()

    def kernel(*args):
        seen.append(args[-1])
        return lambda **kwargs: [output]

    monkeypatch.setattr(decode, "_kernel", kernel)
    assert decode._partials(xh, ids, packed, **options) is output
    assert seen == [expected]


@metal
@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("reuse", [1, 2, 4])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_direct_activation_keeps_scalar_and_vector_outputs(k, reuse, dtype_name):
    import mlx.core as mx
    from exlr8.fast_decode import _expert_mlp

    packed, scales = _fixture(mx, k=k)
    ids = mx.array([[0, 1, 2], [2, 0, 1], [0, 0, 2], [2, 1, 0]], dtype=mx.uint32)
    x = mx.array(np.random.default_rng(619).normal(size=(4, 3, 256)), dtype=getattr(mx, dtype_name))
    config = (1, 4, 1, 4, True, True, True)
    expected = _expert_mlp(x, ids, packed, scales, k, 256, 384, config, reuse=reuse)
    actual = _expert_mlp(x, ids, packed, scales, k, 256, 384, config, reuse=reuse, direct_x=True)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)).view(np.uint32),
                                  np.array(expected.astype(mx.float32)).view(np.uint32))


@metal
@pytest.mark.parametrize("reuse", [1, 4])
def test_direct_activation_keeps_static_cache_outputs(reuse):
    import mlx.core as mx
    from exlr8.fast_decode import _expert_mlp

    packed, scales = _fixture(mx)
    fp8, mapping = _fp8_fixture(mx, packed, 4, 256, 384, [2, 0])
    ids = mx.array([[0, 1, 2], [2, 0, 1], [0, 0, 2], [2, 1, 0]], dtype=mx.uint32)
    x = mx.array(np.random.default_rng(620).normal(size=(4, 256)), dtype=mx.bfloat16)
    config = (1, 4, 1, 4, True, True, True)
    expected = _expert_mlp(x, ids, packed, scales, 4, 256, 384, config, reuse=reuse,
                           fp8=fp8, expert_to_slot=mapping)
    actual = _expert_mlp(x, ids, packed, scales, 4, 256, 384, config, reuse=reuse,
                         fp8=fp8, expert_to_slot=mapping, direct_x=True)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)).view(np.uint32),
                                  np.array(expected.astype(mx.float32)).view(np.uint32))


@metal
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_direct_activation_full_glm_q1_dimensions(dtype_name):
    import mlx.core as mx
    from exlr8.fast_decode import _expert_mlp

    packed, scales = _fixture(mx, hidden=6144, intermediate=2048, experts=9)
    ids = mx.arange(9, dtype=mx.uint32).reshape(1, 9)
    x = mx.array(np.random.default_rng(621).normal(size=(1, 6144)), dtype=getattr(mx, dtype_name))
    expected = _expert_mlp(x, ids, packed, scales, 4, 6144, 2048, (8, 4, 1, 4, True, True, True))
    actual = _expert_mlp(x, ids, packed, scales, 4, 6144, 2048, (8, 4, 1, 4, True, True, True), direct_x=True)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)).view(np.uint32),
                                  np.array(expected.astype(mx.float32)).view(np.uint32))


@metal
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("hidden", [128, 6144])
def test_finish_fused_keeps_typed_weighted_shared_and_residual_sums(dtype_name, hidden):
    import mlx.core as mx
    from exlr8.fast_decode import _finish, _finish_fused

    rng = np.random.default_rng(731)
    dtype = getattr(mx, dtype_name)
    ids = mx.array([7, 0, 3, 3, 6, 2, 5, 1, 8], dtype=mx.uint32)
    inner = mx.array(rng.normal(size=(9, 1, hidden)).astype(np.float32))
    scale = mx.array(rng.uniform(-0.5, 0.5, (9, hidden)).astype(np.float16))
    scores = mx.array(rng.uniform(0.01, 1, (1, 8)).astype(np.float32))
    residual = mx.array(rng.normal(size=(1, hidden)), dtype=dtype)
    values = _finish(inner, ids, scale, hidden, 1, dtype).reshape(1, 9, hidden)
    routed = (values[:, :-1, :] * scores[..., None]).sum(axis=-2).astype(dtype)
    expected = residual + (routed + values[:, -1, :])
    actual = _finish_fused(inner, ids, scale, hidden, scores, residual)
    np.testing.assert_array_equal(np.array(actual.astype(mx.float32)).view(np.uint32),
                                  np.array(expected.astype(mx.float32)).view(np.uint32))


@metal
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_compiled_finish_full_glm_keeps_dependent_states(dtype_name):
    import mlx.core as mx
    from types import SimpleNamespace
    from exlr8.fast_decode import compiled_expert_mlp_finish, fast_expert_mlp

    h, m = 6144, 2048
    packed, scales = _fixture(mx, hidden=h, intermediate=m, experts=9)
    expert = SimpleNamespace(hidden=h, intermediate=m, packed=packed, scales=scales)
    candidate = compiled_expert_mlp_finish(expert)

    def native(x, ids, scores, residual):
        values = fast_expert_mlp(x, ids, packed, scales, 4, h, m)
        routed = (values[..., :-1, :] * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return residual + (routed + values[..., -1, :])

    current = mx.compile(native)
    rng = np.random.default_rng(732)
    dtype = getattr(mx, dtype_name)
    expected = actual = mx.array(rng.normal(size=(1, 1, h)), dtype=dtype)
    norm = mx.array(rng.uniform(0.7, 1.3, h), dtype=dtype)
    for step in range(12):
        ids = mx.array(np.r_[np.roll([7, 0, 3, 3, 6, 2, 5, 1], step), 8].reshape(1, 1, 9), dtype=mx.uint32)
        scores = mx.array(rng.uniform(0.01, 1, (1, 1, 8)).astype(np.float32))
        expected = current(mx.fast.rms_norm(expected, norm, 1e-5), ids, scores, expected)
        actual = candidate(mx.fast.rms_norm(actual, norm, 1e-5), ids, scores, actual)
        mx.eval(expected, actual)
        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)).view(np.uint32),
                                      np.array(expected.astype(mx.float32)).view(np.uint32))
