"""Format checks against the vendored EXL3 reference and Metal operators."""

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

from exlr8.trellis import (
    decode_trellis_reference,
    dequantize_reference,
    hadamard_reference,
    mcg_codebook,
    round_e4m3,
)


def _reference(name):
    source = Path(__file__).resolve().parents[1] / "third_party/ponyexl3/ref" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exl3_reference_{name}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pack = _reference("trellis")
_perm = _reference("perm")
_codebook = _reference("codebook")


def _fixture(k, side, in_features=128, out_features=256, seed=8):
    rng = np.random.default_rng(seed)
    fresh = rng.integers(0, 1 << k, (in_features // 16, out_features // 16, 256), dtype=np.uint16)
    canonical = _pack.pack_trellis(fresh, k)
    codewords = _pack.unpack_trellis(canonical, k)
    values = mcg_codebook()[codewords][..., _perm.tensor_core_perm_inverse()]
    inner = values.reshape(in_features // 16, out_features // 16, 16, 16)
    inner = inner.transpose(0, 2, 1, 3).reshape(in_features, out_features)
    if side == 2:
        stored = canonical.reshape(in_features // 128, 8, out_features // 128, 8, 16 * k).transpose(0, 2, 1, 3, 4)
    else:
        stored = canonical.reshape(in_features // 16, out_features // 128, 8, 16 * k).transpose(1, 0, 2, 3)
    return np.ascontiguousarray(stored).view(np.uint8).reshape(-1), inner


class TrellisFormatTests(unittest.TestCase):
    def test_e4m3_ties_and_saturation(self):
        codes = np.arange(127)
        exponent, fraction = codes >> 3, codes & 7
        values = np.array([f * 2**-9 if e == 0 else (8 + f) * 2.0**(e - 10) for e, f in zip(exponent, fraction)], np.float32)
        midpoints = (values[:-1] + values[1:]) / 2
        expected = values[np.arange(126) + (np.arange(126) & 1)]
        np.testing.assert_array_equal(round_e4m3(midpoints), expected)
        np.testing.assert_array_equal(round_e4m3(-midpoints), -expected)
        np.testing.assert_array_equal(round_e4m3([500, -500, np.inf, -np.inf]), [448, -448, 448, -448])
        self.assertTrue(np.signbit(round_e4m3(-0.0)))

    def test_codebook_matches_exllamav3_half_add(self):
        states = np.arange(0, 65536, 127)
        expected = round_e4m3([_codebook.decode_3inst(int(s), _codebook.CodebookMode.MCG) for s in states])
        np.testing.assert_array_equal(mcg_codebook()[states], expected)
        self.assertEqual(mcg_codebook()[0], 1.875)

    def test_bitstream_permutation_and_both_storage_orders(self):
        for k in (2, 3, 4):
            for side in (0, 1, 2):
                with self.subTest(k=k, side=side):
                    packed, expected = _fixture(k, side, 256, 384)
                    np.testing.assert_array_equal(decode_trellis_reference(packed, k, 256, 384, side), expected)

    def test_hadamard_is_normalized_and_block_local(self):
        x = np.zeros((2, 256), dtype=np.float32)
        x[0, 0], x[1, 129] = 1, 2
        np.testing.assert_allclose(hadamard_reference(hadamard_reference(x)), x, atol=3e-7)
        self.assertTrue(np.all(hadamard_reference(x)[0, 128:] == 0))

    def test_invalid_byte_count(self):
        with self.assertRaisesRegex(ValueError, "byte count"):
            decode_trellis_reference(np.zeros(5, np.uint8), 4, 128, 128)


@unittest.skipUnless(os.environ.get("EXLR8_TEST_METAL") == "1", "set EXLR8_TEST_METAL=1 on a Metal host")
class TrellisMetalTests(unittest.TestCase):
    def test_dense_and_decode(self):
        import mlx.core as mx
        from exlr8.trellis import dequantize_trellis, trellis_linear

        rng = np.random.default_rng(21)
        for k in (2, 3, 4):
            for side in (1, 2):
                with self.subTest(k=k, side=side):
                    packed, _ = _fixture(k, side)
                    suh = rng.uniform(-0.3, 0.3, 128).astype(np.float16)
                    svh = rng.uniform(-0.3, 0.3, 256).astype(np.float16)
                    x = rng.normal(size=(3, 128)).astype(np.float32)
                    weight = dequantize_reference(packed, suh, svh, k, 128, 256, side)
                    actual = trellis_linear(mx.array(x), mx.array(packed), mx.array(suh), mx.array(svh), k, 128, 256, side)
                    decoded = dequantize_trellis(mx.array(packed), mx.array(suh), mx.array(svh), k, 128, 256, side)
                    np.testing.assert_allclose(np.array(decoded), weight, rtol=2e-5, atol=2e-7)
                    np.testing.assert_allclose(np.array(actual), x @ weight.T, rtol=3e-5, atol=3e-6)

    def test_glm_input_width_and_split_reduction(self):
        import mlx.core as mx
        from exlr8.trellis import trellis_linear

        rng = np.random.default_rng(44)
        for k in (3, 4):
            packed, _ = _fixture(k, 1, 6144, 128)
            suh = rng.uniform(-0.1, 0.1, 6144).astype(np.float16)
            svh = rng.uniform(-0.1, 0.1, 128).astype(np.float16)
            x = rng.normal(size=(2, 6144)).astype(np.float32)
            weight = dequantize_reference(packed, suh, svh, k, 6144, 128, 1)
            actual = trellis_linear(mx.array(x), mx.array(packed), mx.array(suh), mx.array(svh), k, 6144, 128, 1)
            np.testing.assert_allclose(np.array(actual), x @ weight.T, rtol=3e-4, atol=7e-6)
        bf16 = mx.array(x).astype(mx.bfloat16)
        result = trellis_linear(bf16, mx.array(packed), mx.array(suh), mx.array(svh), 4, 6144, 128, 1)
        self.assertEqual(result.dtype, mx.bfloat16)
        np.testing.assert_allclose(np.array(result.astype(mx.float32)), np.array(bf16.astype(mx.float32)) @ weight.T, rtol=0.01, atol=0.002)

    def test_expert_gather_and_per_slot_inputs(self):
        import mlx.core as mx
        from exlr8.trellis import expert_linear

        rng = np.random.default_rng(31)
        indices = np.array([[2, 0], [1, 2]], np.uint32)
        for side in (1, 2):
            weights = [_fixture(4, side, seed=e)[0] for e in range(3)]
            packed = np.concatenate(weights)
            suh = rng.uniform(-0.3, 0.3, (3, 128)).astype(np.float16)
            svh = rng.uniform(-0.3, 0.3, (3, 256)).astype(np.float16)
            matrices = np.stack([dequantize_reference(w, suh[e], svh[e], 4, 128, 256, side) for e, w in enumerate(weights)])
            for per_slot in (False, True):
                x = rng.normal(size=(2, 2, 128) if per_slot else (2, 128)).astype(np.float32)
                rows = x if per_slot else np.broadcast_to(x[:, None, :], (2, 2, 128))
                expected = np.einsum("bsi,bsoi->bso", rows, matrices[indices])
                actual = expert_linear(mx.array(x), mx.array(indices), mx.array(packed), mx.array(suh), mx.array(svh), 4, 128, 256, side)
                np.testing.assert_allclose(np.array(actual), expected, rtol=4e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
