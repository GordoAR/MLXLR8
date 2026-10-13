import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from exlr8.dense import DenseLinear, DenseStore, VocabEmbedding
from exlr8.weights import SafeTensorFile


def record(fmt, out, inp, side=1, k=0):
    return (fmt, out, inp, k, side, 128, 0, 0)


def fp8_reference(codes):
    values = []
    for code in range(256):
        exponent, mantissa = (code >> 3) & 15, code & 7
        value = mantissa / 512 if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
        if exponent == 15 and mantissa == 7:
            value = float("nan")
        values.append(-value if code & 128 else value)
    return np.asarray(values, dtype=np.float32)[codes]


def write_file(path, tensors):
    header = {"__metadata__": {"format": "exlr8-v2"}}
    data = bytearray()
    for name, (dtype, array) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [len(data), len(data) + array.nbytes]}
        data.extend(array.tobytes())
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class DenseTests(unittest.TestCase):
    def assert_array_close(self, actual, expected, rtol=1e-5, atol=1e-5):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=rtol, atol=atol)

    def test_safetensors_reads_ranges_and_bf16_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            raw = np.arange(128, dtype=np.uint8).reshape(2, 64)
            bf16 = np.array([0x3F80, 0xC020], dtype=np.uint16)
            write_file(path, {"bytes": ("U8", raw), "norm.weight": ("BF16", bf16)})
            file = SafeTensorFile(path)
            self.assertEqual(file.metadata["format"], "exlr8-v2")
            self.assertEqual(file.raw("bytes").shape, (2, 64))
            self.assert_array_close(file.load_slice("bytes", 64, 16), raw[1, :16])
            self.assert_array_close(file.load("norm.weight").astype(mx.float32), [1, -2.5])
            with self.assertRaises(ValueError):
                file.raw("norm.weight", 1, 2)
            with self.assertRaises(ValueError):
                file.load_slice("bytes", 120, 16)

    def test_mxfp8_all_codes_and_e8m0_scales(self):
        codes = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
        exponents = np.tile(np.arange(120, 128, dtype=np.uint8), (32, 1))
        layer = DenseLinear(record(3, 32, 256), {
            "fp8": mx.array(codes), "e8m0": mx.array(exponents),
        })
        scales = np.exp2(exponents.astype(np.int32) - 127).repeat(32, axis=1)
        self.assert_array_close(layer.dequantize(mx.float32), fp8_reference(codes) * scales, atol=0)

    def test_mxfp8_linear_matches_reference(self):
        rng = np.random.default_rng(20)
        codes = rng.integers(8, 119, (32, 128), dtype=np.uint8)
        codes |= rng.integers(0, 2, codes.shape, dtype=np.uint8) * 128
        exponents = rng.integers(119, 126, (32, 4), dtype=np.uint8)
        x = rng.normal(size=(2, 3, 128)).astype(np.float32)
        layer = DenseLinear(record(3, 32, 128), {
            "fp8": mx.array(codes), "e8m0": mx.array(exponents),
        })
        scales = np.exp2(exponents.astype(np.int32) - 127).repeat(32, axis=1)
        self.assert_array_close(layer(mx.array(x)), x @ (fp8_reference(codes) * scales).T)

    def test_int4_offset_codes_and_input_shard_order(self):
        rng = np.random.default_rng(21)
        codes = rng.integers(0, 16, (32, 256), dtype=np.uint8)
        scales = rng.uniform(0.001, 0.1, (32, 8)).astype(np.float16)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        expected = (codes.astype(np.float32) - 8) * scales.astype(np.float32).repeat(32, axis=1)
        x = rng.normal(size=(2, 256)).astype(np.float32)
        for side in (1, 2):
            if side == 1:
                q4, s4 = packed, scales
            else:
                q4 = np.stack([packed[:, :64], packed[:, 64:]])
                s4 = np.stack([scales[:, :4], scales[:, 4:]])
            layer = DenseLinear(record(2, 32, 256, side), {
                "q4": mx.array(q4), "s4": mx.array(s4),
            })
            with self.subTest(side=side):
                self.assert_array_close(layer.dequantize(mx.float32), expected, atol=0)
                self.assert_array_close(layer(mx.array(x)), x @ expected.T)
                self.assertEqual(layer(mx.array(x, dtype=mx.bfloat16)).dtype, mx.bfloat16)

    def test_block_fp8_scales_on_both_axes(self):
        rng = np.random.default_rng(22)
        codes = rng.integers(8, 119, (256, 256), dtype=np.uint8)
        scales = np.array([[0.03125, 0.012345], [0.00456, 0.07]], dtype=np.float32)
        layer = DenseLinear(record(4, 256, 256), {"fp8": mx.array(codes), "scale": mx.array(scales)})
        expected = fp8_reference(codes) * scales.repeat(128, axis=0).repeat(128, axis=1)
        self.assert_array_close(layer.dequantize(mx.float32), expected, atol=0)

    def test_vocab_scales_and_multidimensional_token_lookup(self):
        rng = np.random.default_rng(23)
        codes = rng.integers(8, 119, (32, 128), dtype=np.uint8)
        scales = rng.uniform(0.0001, 0.003, 32).astype(np.float32)
        tensors = {"fp8": mx.array(codes), "scale": mx.array(scales)}
        head = DenseLinear(record(5, 32, 128), tensors)
        embed = VocabEmbedding(record(5, 32, 128), tensors)
        expected = fp8_reference(codes) * scales[:, None]
        x = rng.normal(size=(2, 128)).astype(np.float32)
        ids = np.array([[31, 0], [2, 2]], dtype=np.int32)
        self.assert_array_close(head(mx.array(x)), x @ expected.T)
        expected_rows = mx.array(expected[ids]).astype(mx.bfloat16).astype(mx.float32)
        self.assert_array_close(embed(mx.array(ids)).astype(mx.float32), np.asarray(expected_rows), atol=0)
        self.assertEqual(embed(mx.array(ids)).shape, (2, 2, 128))

    def test_store_preserves_small_weights_and_module_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            dense = Path(tmp) / "dense"
            dense.mkdir()
            codes = np.full((32, 128), 0x38, dtype=np.uint8)
            correction = np.array([0.25, -0.5], dtype=np.float32)
            write_file(dense / "exlr8-dense-003.safetensors", {
                "self_attn.q_a_proj.record": ("I64", np.array(record(3, 32, 128), dtype=np.int64)),
                "self_attn.q_a_proj.fp8": ("U8", codes),
                "self_attn.q_a_proj.e8m0": ("U8", np.full((32, 4), 127, dtype=np.uint8)),
                "input_layernorm.weight": ("BF16", np.full(128, 0x3F80, dtype=np.uint16)),
                "mlp.gate.e_score_correction_bias": ("F32", correction),
                "__pad.0": ("U8", np.zeros(64, dtype=np.uint8)),
            })
            layer = DenseStore(tmp).layer(3)
            self.assertEqual(set(layer), {"self_attn.q_a_proj", "input_layernorm.weight", "mlp.gate.e_score_correction_bias"})
            self.assert_array_close(layer["mlp.gate.e_score_correction_bias"], correction)
            self.assertEqual(layer["input_layernorm.weight"].dtype, mx.bfloat16)
            self.assertEqual(set(layer["self_attn.q_a_proj"].parameters()), {"packed", "exponents"})

    def test_trellis_linear_matches_decoded_weight(self):
        rng = np.random.default_rng(24)
        x = rng.normal(size=(2, 128)).astype(np.float32)
        layer = DenseLinear(record(1, 128, 128, k=4), {
            "trellis": mx.array(rng.integers(0, 256, (1, 8192), dtype=np.uint8)),
            "suh": mx.array(rng.normal(size=128).astype(np.float16)),
            "svh": mx.array(rng.normal(size=128).astype(np.float16)),
        })
        self.assert_array_close(layer(mx.array(x)), np.asarray(mx.array(x) @ layer.dequantize(mx.float32).T), rtol=2e-5, atol=2e-5)

    def test_unquantized_formats_keep_shipped_precision(self):
        for fmt, dtype, suffix in ((6, mx.bfloat16, "bf16"), (7, mx.float32, "fp32")):
            weight = mx.array([[0.125, -0.75], [0.5, 2]], dtype=dtype)
            layer = DenseLinear(record(fmt, 2, 2), {suffix: weight})
            x = mx.array([[2, -3]], dtype=mx.bfloat16)
            self.assert_array_close(layer(x).astype(mx.float32), [[2.5, -5]])

    def test_bf16_router_keeps_float32_input_and_output(self):
        weight = mx.array([[0.125, -0.75], [0.5, 2]], dtype=mx.bfloat16)
        layer = DenseLinear(record(6, 2, 2), {"bf16": weight})
        x = np.array([[2.0067, -3.0068]], dtype=np.float32)
        self.assert_array_close(layer(mx.array(x)), x @ np.asarray(weight.astype(mx.float32)).T, atol=1e-7)

    def test_dense_trellis_expansion_keeps_exact_e4m3_values(self):
        from exlr8.trellis import decode_trellis_reference

        rng = np.random.default_rng(28)
        for k in (2, 3, 4):
            for side in (1, 2):
                raw = rng.integers(0, 256, (256 * 384 * k // 8,), dtype=np.uint8)
                layer = DenseLinear(record(1, 384, 256, side, k), {
                    "trellis": mx.array(raw), "suh": mx.ones(256), "svh": mx.ones(384),
                })
                actual = mx.dequantize(layer.fp8_packed, scales=layer.exponents, mode="mxfp8", dtype=mx.float32)
                expected = decode_trellis_reference(raw, k, 256, 384, side).T
                with self.subTest(k=k, side=side):
                    self.assert_array_close(actual, expected, rtol=0, atol=0)
                    self.assertNotIn("packed", layer.parameters())

    def test_dense_trellis_optimized_and_packed_paths_keep_signed_scales(self):
        rng = np.random.default_rng(29)
        tensors = {
            "trellis": mx.array(rng.integers(0, 256, (16384,), dtype=np.uint8)),
            "suh": mx.array(rng.normal(size=256).astype(np.float16)),
            "svh": mx.array(rng.normal(size=128).astype(np.float16)),
        }
        optimized = DenseLinear(record(1, 128, 256, side=2, k=4), tensors)
        packed = DenseLinear(record(1, 128, 256, side=2, k=4), tensors, optimize=False)
        x = mx.array(rng.normal(size=(1, 7, 256)).astype(np.float32))
        self.assert_array_close(optimized(x), np.asarray(packed(x)), rtol=3e-5, atol=3e-5)
        self.assert_array_close(optimized.dequantize(mx.float32), np.asarray(packed.dequantize(mx.float32)), rtol=0, atol=0)
        self.assertNotIn("fp8_packed", packed.parameters())


if __name__ == "__main__":
    unittest.main()
