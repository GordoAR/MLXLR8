"""Check packed vocabulary projection and the format/shape dispatch guard."""
import mlx.core as mx
import numpy as np
import pytest

from exlr8.dense import DenseLinear
from exlr8 import vocab_projection


@pytest.mark.parametrize('rows', [1, 4])
@pytest.mark.parametrize('k', [256, 512, 6144])
@pytest.mark.parametrize('strided', [False, True])
def test_complete_projection_matches_decoded_weights(rows, k, strided):
    mx.random.seed(26)
    packed, _ = mx.quantize(mx.random.normal((32, k)), mode='mxfp8')
    exponents = mx.full((32, k//32), 127, mx.uint8)
    scales = mx.linspace(.001, .004, 32)
    values = mx.random.normal((1, rows, k * (2 if strided else 1))).astype(mx.bfloat16)
    x = values[..., ::2] if strided else values
    decoded = mx.dequantize(packed, exponents, mode='mxfp8', dtype=mx.float32)
    expected = ((x.astype(mx.float32) @ decoded.T) * scales).astype(mx.bfloat16)
    actual = vocab_projection.project(x, packed, scales)
    mx.eval(actual, expected)
    assert actual.shape == (1, rows, 32) and actual.dtype == mx.bfloat16
    np.testing.assert_allclose(np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)), rtol=.02, atol=.005)


@pytest.mark.parametrize('rows', [1, 2, 3, 4, 8])
@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('dtype', [mx.bfloat16, mx.float32])
def test_only_supported_vocabulary_shape_uses_shared_path(monkeypatch, rows, batch, optimize, dtype):
    head = DenseLinear((5, 8, 256, 0, 0),
        {'fp8': mx.zeros((8, 256), mx.uint8), 'scale': mx.ones((8,))}, optimize=optimize)
    head.out_features, head.in_features = 154880, 6144
    calls = []
    def shared(x, packed, scales):
        calls.append('shared')
        return mx.zeros((batch, rows, 8), dtype)
    def native(x, packed, **kwargs):
        calls.append('native')
        return mx.zeros((batch, rows, 8), mx.float32)
    monkeypatch.setattr(vocab_projection, 'project', shared)
    monkeypatch.setattr(mx, 'quantized_matmul', native)
    output = head(mx.zeros((batch, rows, 6144), dtype))
    wanted = 'shared' if optimize and batch==1 and rows in (1,4) and dtype==mx.bfloat16 else 'native'
    assert calls == [wanted]
    assert output.shape == (batch, rows, 8) and output.dtype == dtype


def test_other_format_five_dimensions_keep_native_path(monkeypatch):
    head = DenseLinear((5, 8, 256, 0, 0),
        {'fp8': mx.zeros((8, 256), mx.uint8), 'scale': mx.ones((8,))})
    def forbidden(*args):
        raise AssertionError('Other dimensions must keep the native path')
    monkeypatch.setattr(vocab_projection, 'project', forbidden)
    assert head(mx.zeros((1, 4, 256), mx.bfloat16)).shape == (1, 4, 8)
