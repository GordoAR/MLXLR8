"""EXLR8 dense and vocabulary operators for the seven recorded formats."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .weights import SafeTensorFile


_K4_DECODE_RECORDS = {
    "self_attn.q_b_proj": (1, 16384, 2048, 4, 1),
    "self_attn.o_proj": (1, 6144, 16384, 4, 2),
}


class DenseLinear(nn.Module):
    """Apply a stored matrix with selected inference kernels."""

    def __init__(self, record: tuple[int, ...], tensors: dict[str, mx.array], optimize=True,
                 decode_k4=False):
        super().__init__()
        self.record = tuple(int(value) for value in record)
        self.optimize = optimize
        self.format, self.out_features, self.in_features, self.k, self.sharded_side = self.record[:5]
        if self.format == 1:
            self.packed = tensors["trellis"]
            self.suh = tensors["suh"]
            self.svh = tensors["svh"]
            if optimize:
                from .dense_trellis import expand_trellis

                self.fp8_packed, self.exponents = expand_trellis(
                    self.packed, self.k, self.in_features, self.out_features,
                    self.sharded_side,
                )
                if decode_k4 and self.record[:5] in _K4_DECODE_RECORDS.values():
                    self.packed_k4 = self.packed.reshape(-1).view(mx.uint32)
                    if self.sharded_side == 2:
                        from .dense_trellis import encode_mxfp4

                        self.fp4_packed, self.fp4_exponents = encode_mxfp4(self.fp8_packed, self.exponents)
                del self.packed
        elif self.format == 2:
            codes, scales = tensors["q4"], tensors["s4"]
            if self.sharded_side == 2:
                codes = codes.transpose(1, 0, 2).reshape(self.out_features, self.in_features // 2)
                scales = scales.transpose(1, 0, 2).reshape(self.out_features, self.in_features // 32)
            self.packed = codes.view(mx.uint32)
            # MLX affine dequantization calculates in the scale scalar type.
            self.scales = scales.astype(mx.float32)
            self.biases = -8 * self.scales
        elif self.format in (3, 4, 5):
            self.packed = tensors["fp8"].view(mx.uint32)
            if self.format == 3:
                self.exponents = tensors["e8m0"]
            else:
                self.scales = tensors["scale"]
                self.exponents = mx.full(
                    (self.out_features, self.in_features // 32), 127, dtype=mx.uint8
                )
        elif self.format in (6, 7):
            self.weight = tensors["bf16" if self.format == 6 else "fp32"]
        else:
            raise ValueError(f"unsupported EXLR8 dense format {self.format}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features

    def dequantize(self, dtype=mx.bfloat16) -> mx.array:
        """Return the decoded matrix in [output, input] order."""
        if self.format == 1:
            if self.optimize:
                from .trellis import _hadamard

                inner = mx.dequantize(
                    self.fp8_packed, scales=self.exponents, mode="mxfp8", dtype=mx.float32,
                )
                weight = _hadamard(_hadamard(inner.T).T)
                return (weight * self.svh.astype(mx.float32)[:, None]
                        * self.suh.astype(mx.float32)[None, :]).astype(dtype)
            from .trellis import dequantize_trellis

            return dequantize_trellis(
                self.packed, self.suh, self.svh, self.k,
                self.in_features, self.out_features, self.sharded_side,
            ).astype(dtype)
        if self.format == 2:
            return mx.dequantize(
                self.packed, scales=self.scales, biases=self.biases,
                group_size=32, bits=4, dtype=mx.float32,
            ).astype(dtype)
        if self.format in (3, 4, 5):
            weight = mx.dequantize(
                self.packed, scales=self.exponents, mode="mxfp8", dtype=mx.float32
            )
            if self.format == 4:
                weight = weight.reshape(
                    self.out_features // 128, 128, self.in_features // 128, 128
                ) * self.scales[:, None, :, None]
                weight = weight.reshape(self.shape)
            elif self.format == 5:
                weight = weight * self.scales[:, None]
            return weight.astype(dtype)
        return self.weight.astype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        if self.format == 1:
            if self.optimize:
                if ("packed_k4" in self and x.ndim == 3 and x.shape[0] == 1
                        and 1 <= x.shape[1] <= 4 and x.shape[2] == self.in_features
                        and x.dtype == mx.bfloat16):
                    if x.shape[1] == 4:
                        if self.sharded_side == 2:
                            from .dense_trellis import shared_fp4_linear

                            return shared_fp4_linear(x, self.fp4_packed, self.fp4_exponents, self.suh, self.svh)
                        from .dense_trellis import shared_fp8_linear

                        return shared_fp8_linear(x, self.fp8_packed, self.exponents, self.suh, self.svh)
                    from .dense_trellis import k4_linear

                    return k4_linear(x, self.packed_k4, self.exponents, self.suh, self.svh)
                from .dense_trellis import expanded_linear

                return expanded_linear(x, self.fp8_packed, self.exponents, self.suh, self.svh)
            from .trellis import trellis_linear

            return trellis_linear(
                x, self.packed, self.suh, self.svh, self.k,
                self.in_features, self.out_features, self.sharded_side,
            )
        if self.format == 2:
            return mx.quantized_matmul(
                x, self.packed, scales=self.scales, biases=self.biases,
                group_size=32, bits=4,
            ).astype(x.dtype)
        if self.format == 3:
            return mx.quantized_matmul(
                x, self.packed, scales=self.exponents, mode="mxfp8"
            )
        if self.format == 5:
            if (self.optimize and self.shape == (154880, 6144)
                    and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] in (1, 4)
                    and x.shape[2] == 6144 and x.dtype == mx.bfloat16):
                from .vocab_projection import project

                return project(x, self.packed, self.scales)
            # Apply the F32 row scale after an F32 dot product. No scale rounding.
            result = mx.quantized_matmul(
                x.astype(mx.float32), self.packed,
                scales=self.exponents, mode="mxfp8",
            )
            return (result * self.scales).astype(x.dtype)
        if self.format in (6, 7):
            return (x @ self.weight.T).astype(x.dtype)
        return (x @ self.dequantize(x.dtype).T).astype(x.dtype)


class VocabEmbedding(DenseLinear):
    """Decode only the vocabulary rows selected by the token IDs."""

    def __call__(self, token_ids: mx.array) -> mx.array:
        rows = mx.dequantize(
            self.packed[token_ids], scales=self.exponents[token_ids],
            mode="mxfp8", dtype=mx.float32,
        )
        return (rows * self.scales[token_ids, None]).astype(mx.bfloat16)


def _load_file(file: SafeTensorFile, optimize=True, decode_k4=False) -> dict[str, nn.Module | mx.array]:
    result = {}
    matrix_names = [name[:-7] for name in file.tensors if name.endswith(".record")]
    for name in matrix_names:
        record = tuple(file.raw(name + ".record").tolist())
        names = {
            1: ("trellis", "suh", "svh"), 2: ("q4", "s4"),
            3: ("fp8", "e8m0"), 4: ("fp8", "scale"),
            5: ("fp8", "scale"), 6: ("bf16",), 7: ("fp32",),
        }[record[0]]
        tensors = {suffix: file.load(name + "." + suffix) for suffix in names}
        operator = VocabEmbedding if name == "embed_tokens" else DenseLinear
        result[name] = operator(record, tensors, optimize=optimize,
                                decode_k4=decode_k4 and _K4_DECODE_RECORDS.get(name) == record[:5])
    for name in file.tensors:
        if name.endswith((".weight", ".bias", ".e_score_correction_bias")):
            result[name] = file.load(name)
    return result


class DenseStore:
    """Load each dense layer once when the model requests it."""

    def __init__(self, model_path: str | Path, optimize=True, decode_k4=False):
        self.path = Path(model_path) / "dense"
        self.optimize = optimize
        self.decode_k4 = decode_k4

    def layer(self, index: int) -> dict[str, nn.Module | mx.array]:
        return _load_file(SafeTensorFile(self.path / f"exlr8-dense-{index:03d}.safetensors"),
                          self.optimize, self.decode_k4)

    def vocab(self) -> dict[str, nn.Module | mx.array]:
        return _load_file(SafeTensorFile(self.path / "exlr8-vocab.safetensors"), self.optimize)
