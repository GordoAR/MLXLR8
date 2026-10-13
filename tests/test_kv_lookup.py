"""Lookup tables, native arithmetic, and one-token dispatch conditions."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from exlr8 import kv_lookup
from exlr8.dense import DenseLinear
from exlr8.model import Attention


def bf16_bits(values):
    words = values.view(np.uint32)
    hi, lo = words >> 16, words & 65535
    result = (hi + ((lo > 32768) | ((lo == 32768) & ((hi & 1) != 0)))).astype(np.uint16)
    result[np.isnan(values)] = 0x7fc0
    return result


@pytest.fixture(scope="module")
def lookup_weights():
    rng = np.random.default_rng(448)
    codes = rng.integers(0, 256, (28672, 512), dtype=np.uint8)
    codes[(codes & 127) == 127] = 0
    scales = rng.uniform(1e-5, 1e-3, (224, 4)).astype(np.float32)
    operator = DenseLinear(kv_lookup.RECORD, {"fp8": mx.array(codes), "scale": mx.array(scales)})
    packed = kv_lookup.prepare(operator.packed, operator.scales)
    native = operator.dequantize().reshape(64, 448, 512)
    wk, wv = mx.contiguous(native[:, :192]), mx.contiguous(native[:, 192:])
    mx.eval(packed, wk, wv)
    return codes, scales, packed, wk, wv


def test_table_and_split_codes_match_independent_cpu_decode(lookup_weights):
    codes, scales, (wk_codes, wv_codes, table), wk, wv = lookup_weights
    code = np.arange(256, dtype=np.uint16)
    exponent, mantissa = ((code >> 3) & 15).astype(np.int32), (code & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(mantissa, -9),
                      np.ldexp(1 + mantissa / 8, exponent - 7)).astype(np.float32)
    values = np.where(code & 128, -values, values).astype(np.float32)
    values[(code & 127) == 127] = np.nan
    expected = bf16_bits(scales.reshape(-1, 1) * values[None, :])
    np.testing.assert_array_equal(np.asarray(table.view(mx.uint16)), expected)
    full_codes = codes.reshape(64, 448, 512)
    np.testing.assert_array_equal(np.asarray(wk_codes), full_codes[:, :192])
    np.testing.assert_array_equal(np.asarray(wv_codes), full_codes[:, 192:])
    for offset, rows, weight in ((0, 192, wk), (192, 256, wv)):
        original_rows = np.arange(64)[:, None, None] * 448 + offset + np.arange(rows)[None, :, None]
        bits = expected[(original_rows // 128) * 4 + np.arange(512)[None, None, :] // 128,
                        full_codes[:, offset:offset + rows]]
        np.testing.assert_array_equal(np.asarray(weight.view(mx.uint16)), bits)


def test_both_projection_outputs_match_native_bits(lookup_weights):
    _, _, (wk_codes, wv_codes, table), wk, wv = lookup_weights
    rng = np.random.default_rng(256)
    full_query = mx.array(rng.normal(size=(1, 64, 1, 256)).astype(np.float32)).astype(mx.bfloat16)
    value = mx.array(rng.normal(size=(1, 64, 1, 512)).astype(np.float32)).astype(mx.bfloat16)
    actual = (kv_lookup.query_projection(full_query, wk_codes, table),
              kv_lookup.value_projection(value, wv_codes, table))
    expected = (full_query[..., :192] @ wk, value @ wv.swapaxes(-1, -2))
    mx.eval(actual, expected)
    for left, right in zip(actual, expected):
        assert bool(mx.all(mx.isfinite(left)).item())
        np.testing.assert_array_equal(np.asarray(left.view(mx.uint16)), np.asarray(right.view(mx.uint16)))


@pytest.mark.parametrize("shape,dtype,enabled,use_lookup", [
    ((1, 64, 1, 256), mx.bfloat16, True, True),
    ((1, 64, 2, 256), mx.bfloat16, True, False),
    ((2, 64, 1, 256), mx.bfloat16, True, False),
    ((1, 64, 1, 256), mx.float32, True, False),
    ((1, 64, 1, 256), mx.bfloat16, False, False),
])
def test_query_dispatch_keeps_full_head_storage(monkeypatch, shape, dtype, enabled, use_lookup):
    class Value:
        def __matmul__(self, weight):
            return "native"
    full_query = SimpleNamespace(shape=shape, dtype=dtype)
    attention = SimpleNamespace(_lookup=enabled, wk=object(), wk_codes=object(), kv_lookup_table=object())
    def selected(x, weight, table):
        assert x is full_query and weight is attention.wk_codes and table is attention.kv_lookup_table
        return "lookup"
    monkeypatch.setattr(kv_lookup, "query_projection", selected)
    assert Attention._absorbed_queries(attention, Value(), full_query) == ("lookup" if use_lookup else "native")


@pytest.mark.parametrize("shape,dtype,enabled,use_lookup", [
    ((1, 64, 1, 512), mx.bfloat16, True, True),
    ((1, 64, 2, 512), mx.bfloat16, True, False),
    ((2, 64, 1, 512), mx.bfloat16, True, False),
    ((1, 64, 1, 512), mx.float32, True, False),
    ((1, 64, 1, 512), mx.bfloat16, False, False),
])
def test_value_dispatch_requires_one_bf16_token(monkeypatch, shape, dtype, enabled, use_lookup):
    class Weight:
        def swapaxes(self, *_):
            return self
    class Value:
        def __matmul__(self, weight):
            return "native"
    value = Value()
    value.shape, value.dtype = shape, dtype
    attention = SimpleNamespace(_lookup=enabled, wv=Weight(), wv_codes=object(), kv_lookup_table=object())
    def selected(x, weight, table):
        assert x is value and weight is attention.wv_codes and table is attention.kv_lookup_table
        return "lookup"
    monkeypatch.setattr(kv_lookup, "value_projection", selected)
    assert Attention._project_values(attention, value) == ("lookup" if use_lookup else "native")
