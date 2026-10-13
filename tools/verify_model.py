"""Compare the real dense backbone with public MLX or oMLX decoder layers.

Run from the project root:
    .venv/bin/python tools/verify_model.py --reference mlx --tokens 11
    .venv/bin/python tools/verify_model.py --reference omlx --tokens 5
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_tokenizer

from exlr8.dense import DenseLinear, DenseStore
from exlr8.model import DecoderLayer, ModelArgs


class DecodedLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x @ self.weight.T

    def dequantize(self):
        return self.weight


def public_layer(args, index, weights, attention, layer_type):
    layer = layer_type(args, index)
    for name in ("q_a_proj", "q_b_proj", "o_proj"):
        setattr(layer.self_attn, name, weights["self_attn." + name])
    layer.self_attn.kv_a_proj_with_mqa = weights["self_attn.kv_a_proj_with_mqa"]
    layer.self_attn.q_a_layernorm.weight = weights["self_attn.q_a_layernorm.weight"]
    layer.self_attn.kv_a_layernorm.weight = weights["self_attn.kv_a_layernorm.weight"]
    layer.self_attn.embed_q.weight = mx.contiguous(attention.wk.swapaxes(-1, -2))
    layer.self_attn.unembed_out.weight = attention.wv
    for name in ("wq_b", "wk", "weights_proj"):
        setattr(layer.self_attn.indexer, name, weights["self_attn.indexer." + name])
    layer.self_attn.indexer.k_norm.weight = weights["self_attn.indexer.k_norm.weight"]
    layer.self_attn.indexer.k_norm.bias = weights["self_attn.indexer.k_norm.bias"]
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(layer.mlp, name, weights["mlp." + name])
    layer.input_layernorm.weight = weights["input_layernorm.weight"]
    layer.post_attention_layernorm.weight = weights["post_attention_layernorm.weight"]
    return layer


def verify(args):
    if args.reference == "omlx":
        sys.path.insert(0, str(args.omlx_path))
        from omlx.patches.glm_moe_dsa.glm_moe_dsa_model import (
            GlmMoeDsaDecoderLayer as ReferenceLayer,
            ModelArgs as ReferenceArgs,
        )
    else:
        if 1 < args.tokens <= 8:
            raise ValueError("Use one or more than eight tokens for the MLX reference; use oMLX for 2..8")
        from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer as ReferenceLayer
        from mlx_lm.models.glm_moe_dsa import ModelArgs as ReferenceArgs

    config = json.loads((args.model / "config.json").read_text())
    model_args, reference_args = ModelArgs.from_dict(config), ReferenceArgs.from_dict(config)
    store = DenseStore(args.model)
    tokenizer = load_tokenizer(args.model, eos_token_ids=config["eos_token_id"])
    token_ids = tokenizer.encode(args.prompt)
    token_ids = (token_ids * ((args.tokens + len(token_ids) - 1) // len(token_ids)))[:args.tokens]
    x = store.vocab()["embed_tokens"](mx.array([token_ids]))
    mask = mx.arange(args.tokens)[:, None] >= mx.arange(args.tokens)[None, :] if args.tokens > 1 else None
    results = []
    for index in range(model_args.first_k_dense_replace):
        start = time.perf_counter()
        packed = store.layer(index)
        decoded = {name: DecodedLinear(value.dequantize()) if isinstance(value, DenseLinear) else value
                   for name, value in packed.items()}
        own = DecoderLayer(model_args, index, decoded)
        reference = public_layer(reference_args, index, decoded, own.self_attn, ReferenceLayer)
        actual, _ = own(x, mask)
        expected = reference(x, mask)
        if args.reference == "omlx":
            expected, _ = expected
        packed_output, _ = DecoderLayer(model_args, index, packed)(x, mask)
        mx.eval(actual, expected, packed_output)
        reference_error = actual.astype(mx.float32) - expected.astype(mx.float32)
        packed_error = packed_output.astype(mx.float32) - expected.astype(mx.float32)
        rms = float(mx.sqrt(mx.mean(expected.astype(mx.float32)**2)).item())
        results.append({
            "layer": index,
            "architecture_max_error": float(mx.max(mx.abs(reference_error)).item()),
            "packed_operator_rms_error": float(mx.sqrt(mx.mean(packed_error**2)).item()),
            "reference_rms": rms,
            "finite": bool(mx.all(mx.isfinite(packed_output)).item()),
            "seconds": round(time.perf_counter() - start, 3),
        })
        x = expected
        del packed, decoded, own, reference, actual, expected, packed_output
        mx.clear_cache()
    return {
        "model": str(args.model), "reference": args.reference,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "input_token_ids": token_ids, "dtype": "bfloat16", "layers": results,
        "architecture_bit_identical": all(r["architecture_max_error"] == 0 for r in results),
        "all_finite": all(r["finite"] for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / ".lmstudio/models/kindlingai/glm-5.3-exlr8-up1")
    parser.add_argument("--reference", choices=("mlx", "omlx"), default="mlx")
    parser.add_argument("--omlx-path", type=Path, default=Path("/Applications/oMLX.app/Contents/Resources"))
    parser.add_argument("--tokens", type=int, default=11)
    parser.add_argument("--prompt", default="The capital of France is Paris. This is a model test.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    result = verify(args)
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")
    if not result["architecture_bit_identical"] or not result["all_finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
