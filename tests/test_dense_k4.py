"""Exact K4 decode and the conditions which select it."""

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from exlr8 import dense, dense_trellis


RECORDS = [(1, 16384, 2048, 4, 1), (1, 6144, 16384, 4, 2)]


def fake_expansion(monkeypatch):
    monkeypatch.setattr(dense_trellis, "expand_trellis", lambda *args: (
        mx.zeros((1, 1), dtype=mx.uint32), mx.full((1, 1), 127, dtype=mx.uint8)))
    return {"trellis": mx.array([1, 2, 3, 4], dtype=mx.uint8),
            "suh": mx.ones((1,)), "svh": mx.ones((1,))}


@pytest.mark.parametrize("record,enabled,optimize,retained", [
    (RECORDS[0], True, True, True), (RECORDS[1], True, True, True),
    (RECORDS[0], False, True, False), (RECORDS[0], True, False, False),
    ((1, 16384, 2048, 3, 1), True, True, False),
    ((1, 16384, 2048, 4, 2), True, True, False),
    ((1, 6144, 16384, 4, 1), True, True, False),
    ((1, 128, 256, 4, 2), True, True, False),
])
def test_k4_retention_requires_supported_record(monkeypatch, record, enabled, optimize, retained):
    tensors = fake_expansion(monkeypatch)
    layer = dense.DenseLinear(record, tensors, optimize=optimize, decode_k4=enabled)
    assert ("packed_k4" in layer) == retained
    assert ("packed_k4" in layer.parameters()) == retained
    if retained:
        assert layer.packed_k4.dtype == mx.uint32
        np.testing.assert_array_equal(np.asarray(layer.packed_k4.view(mx.uint8)), [1, 2, 3, 4])
        assert "fp8_packed" in layer.parameters()
        assert "exponents" in layer.parameters()


@pytest.mark.parametrize("shape,dtype,use_k4", [
    ((1, 1, 2048), mx.bfloat16, True),
    ((1, 2, 2048), mx.bfloat16, False),
    ((2, 1, 2048), mx.bfloat16, False),
    ((1, 2048), mx.bfloat16, False),
    ((2048,), mx.bfloat16, False),
    ((1, 1, 2048), mx.float16, False),
    ((1, 1, 2048), mx.float32, False),
])
def test_k4_dispatch_requires_one_bf16_token(monkeypatch, shape, dtype, use_k4):
    layer = dense.DenseLinear(RECORDS[0], fake_expansion(monkeypatch), decode_k4=True)
    calls = []
    def selected(*args):
        calls.append("k4")
        assert args[1] is layer.packed_k4
        return args[0]
    def fallback(*args):
        calls.append("fp8")
        assert args[1] is layer.fp8_packed
        return args[0]
    monkeypatch.setattr(dense_trellis, "k4_linear", selected)
    monkeypatch.setattr(dense_trellis, "expanded_linear", fallback)
    x = mx.zeros(shape, dtype=dtype)
    assert layer(x) is x
    assert calls == (["k4"] if use_k4 else ["fp8"])


@pytest.mark.parametrize("name,record,expected", [
    ("self_attn.q_b_proj", RECORDS[0], True),
    ("self_attn.o_proj", RECORDS[1], True),
    ("self_attn.q_a_proj", RECORDS[0], False),
    ("self_attn.q_b_proj", RECORDS[1], False),
    ("self_attn.o_proj", RECORDS[0], False),
])
def test_file_loader_checks_name_and_record(monkeypatch, name, record, expected):
    calls = []
    monkeypatch.setattr(dense, "DenseLinear", lambda rec, tensors, **kwargs: calls.append(kwargs))
    file = SimpleNamespace(tensors=[name + ".record"],
                           raw=lambda key: np.array(record), load=lambda key: None)
    dense._load_file(file, decode_k4=True)
    assert calls == [{"optimize": True, "decode_k4": expected}]


@pytest.mark.parametrize("enabled,optimize,resident,expert_optimize,expected", [
    (True, True, True, True, True), (False, True, True, True, False),
    (True, False, True, True, False), (True, True, False, True, False),
    (True, True, True, False, False),
])
def test_model_loader_requires_resident_optimized_experts(
        monkeypatch, tmp_path, enabled, optimize, resident, expert_optimize, expected):
    from exlr8 import model
    calls = []
    class Store:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
        def vocab(self):
            return {}
    monkeypatch.setattr(dense, "DenseStore", Store)
    monkeypatch.setattr(model.ModelArgs, "from_dict", lambda data: SimpleNamespace(num_hidden_layers=0))
    monkeypatch.setattr(model, "Model", lambda *args: SimpleNamespace(parameters=lambda: []))
    (tmp_path / "config.json").write_text("{}")
    expert = SimpleNamespace(resident=resident, optimize=expert_optimize)
    model.load_model(tmp_path, expert, optimize=optimize, decode_k4=enabled)
    assert calls == [{"optimize": optimize, "decode_k4": expected}]


@pytest.mark.parametrize("resident,no_optimize,mtp,expected", [
    (True, False, False, True), (False, False, False, False),
    (True, True, False, False), (True, False, True, False),
])
def test_cli_disables_k4_for_mtp_and_nonresident_mode(
        monkeypatch, tmp_path, resident, no_optimize, mtp, expected):
    from exlr8 import cli
    calls = []
    model = SimpleNamespace(parameters=lambda: [])
    def load(*args, **kwargs):
        calls.append(kwargs)
        return model
    monkeypatch.setattr(cli, "maybe_set_recommended_wired_limit", lambda: None)
    monkeypatch.setattr(mx, "set_cache_limit", lambda value: None)
    monkeypatch.setattr(mx, "set_memory_limit", lambda value: None)
    monkeypatch.setattr(cli, "ExpertStore", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(cli, "load_model", load)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "MTPRuntime", lambda *args, **kwargs: model)
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 0}))
    args = SimpleNamespace(model=tmp_path, memory_limit_gib=1, expert_cache_gib=0,
                           resident=resident, no_optimize=no_optimize, mtp=mtp)
    cli.load_runtime(args)
    assert calls[0]["decode_k4"] is expected


@pytest.mark.parametrize("record", RECORDS)
def test_k4_real_shapes_match_native_inner_and_signed_full_output(record):
    _, n, k, _, side = record
    rng = np.random.default_rng(2048 + side)
    tensors = {"trellis": mx.array(rng.integers(0, 256, n * k // 2, dtype=np.uint8)),
               "suh": mx.array(rng.uniform(-1.5, 1.5, k).astype(np.float16)),
               "svh": mx.array(rng.uniform(-1.5, 1.5, n).astype(np.float16))}
    layer = dense.DenseLinear(record, tensors, decode_k4=True)
    x = mx.array(rng.normal(size=(1, 1, k)).astype(np.float32)).astype(mx.bfloat16)
    xh = dense_trellis._hadamard(x.astype(mx.float32) * layer.suh.astype(mx.float32))
    actual_inner = dense_trellis._k4_kernel(k, n)(
        inputs=[xh.reshape(1, k), layer.packed_k4, layer.exponents],
        grid=(32, n // 8 * 2, 1), threadgroup=(32, 2, 1),
        output_shapes=[(1, n)], output_dtypes=[mx.float32])[0]
    expected_inner = mx.quantized_matmul(
        xh.reshape(1, k), layer.fp8_packed, scales=layer.exponents, mode="mxfp8")
    actual = layer(x)
    expected = dense_trellis.expanded_linear(
        x, layer.fp8_packed, layer.exponents, layer.suh, layer.svh)
    mx.eval(actual_inner, expected_inner, actual, expected)
    parameters = layer.parameters()
    assert {"packed_k4", "fp8_packed", "exponents"} <= parameters.keys()
    assert layer.packed_k4.nbytes == n * k // 2
    assert mx.get_array_buffer_size(layer.packed_k4) == layer.packed_k4.nbytes
    native = [layer.fp8_packed, layer.exponents]
    assert mx.get_array_buffer_size(layer.packed_k4, native) == (
        layer.packed_k4.nbytes + mx.get_array_buffer_size(native))
    assert bool(mx.all(mx.isfinite(actual_inner)).item())
    np.testing.assert_array_equal(np.asarray(actual_inner.view(mx.uint32)),
                                  np.asarray(expected_inner.view(mx.uint32)))
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint16)),
                                  np.asarray(expected.view(mx.uint16)))
    assert dense_trellis._k4_kernel.cache_info().maxsize == 2
