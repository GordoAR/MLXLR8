"""GLM-5.3 inference with packed EXLR8 weights and latent MLA caches.

The attention and router equations follow the public MLX-LM/oMLX GLM DSA
implementation. Dense and expert weights are supplied as packed callables;
constructing this model never creates random full-precision expert matrices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "glm_moe_dsa"
    hidden_size: int = 6144
    vocab_size: int = 154880
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    first_k_dense_replace: int = 3
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    rope_parameters: dict | None = None
    rope_interleave: bool = True
    indexer_rope_interleave: bool = True
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    indexer_types: list[str] | None = None
    index_topk_pattern: list[str] | str | None = None
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3
    mlp_layer_types: list[str] | None = None

    def __post_init__(self):
        if self.model_type != "glm_moe_dsa":
            raise ValueError("This loader requires a glm_moe_dsa checkpoint")
        if (self.hidden_act, self.scoring_func, self.topk_method) != (
            "silu", "sigmoid", "noaux_tc"
        ):
            raise ValueError("Unsupported GLM activation or router")
        if self.n_shared_experts != 1:
            raise ValueError("EXLR8 GLM requires one shared expert")
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 8000000}
        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern]
                    if isinstance(pattern, str) else list(pattern)
                )
            else:
                self.indexer_types = [
                    "full" if max(i - self.index_skip_topk_offset + 1, 0)
                    % max(self.index_topk_freq, 1) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse"
                for i in range(self.num_hidden_layers)
            ]
        for name, allowed in (("indexer_types", {"full", "shared"}),
                              ("mlp_layer_types", {"dense", "sparse"})):
            values = getattr(self, name)
            if len(values) != self.num_hidden_layers or not set(values) <= allowed:
                raise ValueError(f"Invalid {name} for {self.num_hidden_layers} layers")
        if self.indexer_types[0] != "full":
            raise ValueError("The first layer must have a full indexer")


def _rope(args: ModelArgs, traditional: bool):
    return initialize_rope(
        dims=args.qk_rope_head_dim,
        base=args.rope_parameters["rope_theta"],
        traditional=traditional,
        max_position_embeddings=args.max_position_embeddings,
        scaling_config=args.rope_parameters,
    )


def route_experts(logits, bias, args: ModelArgs):
    """Select with corrected scores; weight with the original sigmoid scores."""
    scores = mx.sigmoid(logits.astype(mx.float32))
    choice = scores + bias
    if args.topk_group < args.n_group:
        grouped = mx.unflatten(choice, -1, (args.n_group, -1))
        group_scores = mx.topk(grouped, 2, axis=-1).sum(axis=-1)
        groups = mx.argpartition(-group_scores, args.topk_group - 1, axis=-1)[
            ..., :args.topk_group
        ]
        keep = mx.put_along_axis(
            mx.zeros(group_scores.shape, dtype=mx.bool_), groups,
            mx.array(True), axis=-1,
        )
        choice = mx.flatten(mx.where(keep[..., None], grouped, -float("inf")), -2)
    indices = mx.argpartition(-choice, args.num_experts_per_tok - 1, axis=-1)[
        ..., :args.num_experts_per_tok
    ]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if args.norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return indices.astype(mx.uint32), weights * args.routed_scaling_factor


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, weights: dict):
        super().__init__()
        self.args = args
        prefix = "self_attn.indexer."
        self.wq_b = weights[prefix + "wq_b"]
        self.wk = weights[prefix + "wk"]
        self.weights_proj = weights[prefix + "weights_proj"]
        self.norm_weight = weights[prefix + "k_norm.weight"]
        self.norm_bias = weights[prefix + "k_norm.bias"]
        self.rope = _rope(args, args.indexer_rope_interleave)

    def __call__(self, x, qr, mask, cache=None):
        a = self.args
        batch, length, _ = x.shape
        k = mx.fast.layer_norm(self.wk(x), self.norm_weight, self.norm_bias, 1e-6)
        k = k[:, None, :, :]
        offset = cache.offset if cache is not None else 0
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((batch, 1, length, 0), k.dtype))
        if k.shape[2] <= a.index_topk:
            return None
        q = self.wq_b(qr).reshape(batch, length, a.index_n_heads, a.index_head_dim)
        q = self.rope(q.swapaxes(1, 2), offset=offset)
        head_weights = self.weights_proj(x).astype(mx.float32).swapaxes(1, 2)
        head_weights *= a.index_n_heads**-0.5
        q, k = q.astype(mx.float32), k.astype(mx.float32).swapaxes(-1, -2)
        selections = []
        # Bound the index score temporary during long-context prefill.
        chunk = min(64, max(1, (64 << 20) // (batch * a.index_n_heads * k.shape[-1] * 4)))
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            scores = q[:, :, start:end, :] @ k
            scores = mx.maximum(scores * a.index_head_dim**-0.5, 0)
            scores = (scores * head_weights[:, :, start:end, None]).sum(axis=1, keepdims=True)
            if mask is not None:
                scores = mx.where(mask[..., start:end, :], scores, -float("inf"))
            selections.append(mx.argpartition(-scores, a.index_topk - 1, axis=-1)[..., :a.index_topk])
        return mx.concatenate(selections, axis=2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer: int, weights: dict):
        super().__init__()
        self.args = args
        self.q_a_proj = weights["self_attn.q_a_proj"]
        self.q_b_proj = weights["self_attn.q_b_proj"]
        self.kv_a_proj = weights["self_attn.kv_a_proj_with_mqa"]
        self._joined_input = None
        from .dense import DenseLinear

        qa, kva = self.q_a_proj, self.kv_a_proj
        if (layer < args.num_hidden_layers and "eh_proj" not in weights
                and isinstance(qa, DenseLinear) and isinstance(kva, DenseLinear)
                and qa.optimize and kva.optimize and qa.format == kva.format == 3
                and qa.in_features == kva.in_features
                and qa.out_features % 8 == 0 and kva.out_features % 8 == 0):
            packed = mx.concatenate((qa.packed, kva.packed), axis=0)
            exponents = mx.concatenate((qa.exponents, kva.exponents), axis=0)
            rows = qa.out_features
            qa.packed, kva.packed = packed[:rows], packed[rows:]
            qa.exponents, kva.exponents = exponents[:rows], exponents[rows:]

            # The row views remain parameters; the closure adds no parameter bytes.
            def joined_input(x):
                value = mx.quantized_matmul(x, packed, scales=exponents, mode="mxfp8")
                return mx.split(value, [rows], axis=-1)

            self._joined_input = joined_input
        self.o_proj = weights["self_attn.o_proj"]
        self.q_norm = weights["self_attn.q_a_layernorm.weight"]
        self.kv_norm = weights["self_attn.kv_a_layernorm.weight"]
        kv = weights["self_attn.kv_b_proj"].dequantize().reshape(
            args.num_attention_heads, args.qk_nope_head_dim + args.v_head_dim,
            args.kv_lora_rank,
        )
        self.wk = mx.contiguous(kv[:, :args.qk_nope_head_dim, :])
        self.wv = mx.contiguous(kv[:, args.qk_nope_head_dim:, :])
        self._lookup = False
        self.indexer = Indexer(args, weights) if args.indexer_types[layer] == "full" else None
        self.rope = _rope(args, args.rope_interleave)
        self.scale = (args.qk_nope_head_dim + args.qk_rope_head_dim)**-0.5
        parameters = args.rope_parameters
        if parameters.get("mscale_all_dim", 0) and parameters.get("factor", 1) > 1:
            import math
            self.scale *= (0.1 * parameters["mscale_all_dim"] * math.log(parameters["factor"]) + 1)**2

    def _absorbed_queries(self, q_nope, full_query=None):
        if (self._lookup and full_query is not None
                and len(full_query.shape) == 4 and full_query.shape[:2] == (1, 64)
                and 1 <= full_query.shape[2] <= 4 and full_query.shape[3] == 256
                and full_query.dtype == mx.bfloat16):
            from .kv_lookup import query_projection

            return query_projection(full_query, self.wk_codes, self.kv_lookup_table)
        return q_nope @ self.wk

    def _project_values(self, output):
        if (self._lookup and len(output.shape) == 4 and output.shape[:2] == (1, 64)
                and 1 <= output.shape[2] <= 4 and output.shape[3] == 512
                and output.dtype == mx.bfloat16):
            from .kv_lookup import value_projection

            return value_projection(output, self.wv_codes, self.kv_lookup_table)
        return output @ self.wv.swapaxes(-1, -2)

    def _sparse_attention(self, q, q_pe, latent, k_pe, indices, mask):
        batch, heads, length, _ = q.shape
        key_length, selected = latent.shape[2], indices.shape[-1]
        outputs = []
        # Treat each sparse query as a decode row to avoid full expanded K/V.
        row_bytes = batch * selected * (
            (latent.shape[-1] + k_pe.shape[-1]) * latent.dtype.size
            + heads * (8 + q.dtype.size)
        )
        chunk = min(32, max(1, (128 << 20) // row_bytes))
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            rows = end - start
            idx = indices[:, 0, start:end, :]
            def gather(source):
                dims = source.shape[-1]
                return mx.take_along_axis(
                    mx.broadcast_to(source, (batch, rows, key_length, dims)),
                    mx.broadcast_to(idx[..., None], (batch, rows, selected, dims)), axis=2,
                ).reshape(batch * rows, 1, selected, dims)
            lat_rows, pe_rows = gather(latent), gather(k_pe)
            q_rows = q[:, :, start:end, :].swapaxes(1, 2).reshape(batch * rows, heads, 1, -1)
            qp_rows = q_pe[:, :, start:end, :].swapaxes(1, 2).reshape(batch * rows, heads, 1, -1)
            pe_scores = (qp_rows * self.scale) @ pe_rows.swapaxes(-1, -2)
            valid = idx <= mx.arange(key_length - length + start, key_length - length + end)[None, :, None]
            if mask is not None:
                row_mask = mx.broadcast_to(mask, (batch, 1, length, key_length))[:, 0, start:end, :]
                valid &= mx.take_along_axis(row_mask, idx, axis=-1)
            pe_scores = mx.where(valid.reshape(batch * rows, 1, 1, selected), pe_scores, -float("inf"))
            # Put the heads in GEMM rows so they reuse each sparse K/V matrix.
            q_rows = q_rows.swapaxes(1, 2)
            pe_scores = pe_scores.swapaxes(1, 2)
            output = mx.fast.scaled_dot_product_attention(
                q_rows, lat_rows, lat_rows, scale=self.scale, mask=pe_scores,
            )
            outputs.append(output.reshape(batch, rows, heads, -1).swapaxes(1, 2))
        return mx.concatenate(outputs, axis=2)

    def __call__(self, x, mask=None, cache=None, prev_topk_indices=None):
        a = self.args
        batch, length, _ = x.shape
        joined = (self._joined_input(x) if self._joined_input is not None
                  and batch == length == 1 and x.dtype == mx.bfloat16 else None)
        qr = mx.fast.rms_norm(self.q_a_proj(x) if joined is None else joined[0], self.q_norm, 1e-6)
        q = self.q_b_proj(qr).reshape(batch, length, a.num_attention_heads, -1).swapaxes(1, 2)
        q_nope, q_pe = mx.split(q, [a.qk_nope_head_dim], axis=-1)
        latent, k_pe = mx.split(self.kv_a_proj(x) if joined is None else joined[1], [a.kv_lora_rank], axis=-1)
        latent = mx.fast.rms_norm(latent, self.kv_norm, 1e-6)[:, None, :, :]
        k_pe = k_pe[:, None, :, :]
        offset = cache[0].offset if cache is not None else 0
        q_pe, k_pe = self.rope(q_pe, offset=offset), self.rope(k_pe, offset=offset)
        if cache is not None:
            latent, k_pe = cache[0].update_and_fetch(latent, k_pe)
        topk = (
            self.indexer(x, qr, mask, cache[1] if cache is not None else None)
            if self.indexer is not None else prev_topk_indices
        )
        if self.indexer is not None and cache is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))
        if topk is not None:
            output = self._sparse_attention(self._absorbed_queries(q_nope, q), q_pe, latent, k_pe, topk, mask)
            output = self._project_values(output)
        else:
            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(mask, pe_scores, -float("inf"))
            absorbed = length <= 8 or latent.shape[2] >= 4 * length
            queries = self._absorbed_queries(q_nope, q) if absorbed else q_nope
            keys = latent if absorbed else latent @ self.wk.swapaxes(-1, -2)
            values = latent if absorbed else latent @ self.wv.swapaxes(-1, -2)
            output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=pe_scores)
            if absorbed:
                output = self._project_values(output)
        return self.o_proj(output.swapaxes(1, 2).reshape(batch, length, -1)), topk


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, index: int, weights: dict, experts=None):
        super().__init__()
        self.args = args
        self.self_attn = Attention(args, index, weights)
        self.input_norm = weights["input_layernorm.weight"]
        self.post_norm = weights["post_attention_layernorm.weight"]
        self.experts = experts
        self._expert_finish = None
        self._fused_rms = False
        if args.mlp_layer_types[index] == "dense":
            self.gate_proj = weights["mlp.gate_proj"]
            self.up_proj = weights["mlp.up_proj"]
            self.down_proj = weights["mlp.down_proj"]
        else:
            if experts is None:
                raise ValueError(f"Missing packed experts for layer {index}")
            from .dense import DenseLinear

            self.gate = weights["mlp.gate"]
            if (index < args.num_hidden_layers and "eh_proj" not in weights
                    and isinstance(self.gate, DenseLinear) and self.gate.optimize
                    and self.gate.format == 6):
                self.gate.weight = self.gate.weight.astype(mx.float32)
            self.bias = weights["mlp.gate.e_score_correction_bias"]

    def __call__(self, x, mask=None, cache=None, prev_topk_indices=None):
        a = self.args
        r, topk = self.self_attn(mx.fast.rms_norm(x, self.input_norm, a.rms_norm_eps), mask, cache, prev_topk_indices)
        if (self._fused_rms and x.shape == r.shape and x.ndim == 3 and x.shape[0] == 1
                and 1 <= x.shape[1] <= 4 and x.shape[2] == 6144
                and x.dtype == r.dtype == mx.bfloat16):
            from .fast_norm import residual_rms_pair

            h, x = residual_rms_pair(x, r, self.post_norm)
        else:
            h = x + r
            x = mx.fast.rms_norm(h, self.post_norm, a.rms_norm_eps)
        if self.experts is None:
            r = self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
        else:
            ids, scores = route_experts(self.gate(x.astype(mx.float32)), self.bias, a)
            shared = mx.full(ids.shape[:-1] + (1,), a.n_routed_experts, dtype=mx.uint32)
            indices = mx.concatenate([ids, shared], axis=-1)
            if (1 <= x.size // x.shape[-1] <= 4 and a.num_experts_per_tok == 8 and a.n_shared_experts == 1
                    and getattr(self.experts, "k", None) == 4
                    and self.experts.store.resident and self.experts.store.optimize):
                if self._expert_finish is None:
                    from .fast_decode import compiled_expert_mlp_finish

                    self._expert_finish = compiled_expert_mlp_finish(self.experts)
                return self._expert_finish(x, indices, scores, h), topk
            values = self.experts(x, indices)
            r = (values[..., :-1, :] * scores[..., None]).sum(axis=-2).astype(x.dtype)
            r = r + values[..., -1, :]
        return h + r, topk


class Model(nn.Module):
    def __init__(self, args: ModelArgs, layers: list[DecoderLayer], vocab: dict):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.layers = layers
        self.embed_tokens = vocab["embed_tokens"]
        self.lm_head = vocab["lm_head"]
        self.norm = vocab["norm.weight"]
        self.trace = None
        self.freeze()

    def make_cache(self):
        return [CacheList(KVCache(), KVCache()) if layer.self_attn.indexer is not None
                else CacheList(KVCache()) for layer in self.layers]

    def forward_hidden(self, inputs, cache=None):
        """Return backbone states before the final norm and shared DSA indices."""
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("The cache must contain one entry per model layer")
        h = self.embed_tokens(inputs)
        mask = create_attention_mask(h, cache[0][0] if cache is not None else None, return_array=True)
        topk = None
        for index, layer in enumerate(self.layers):
            h, topk = layer(h, mask, cache[index] if cache is not None else None, topk)
            if self.trace is not None:
                self.trace(index, h)
        return h, topk

    def project_hidden(self, hidden):
        return self.lm_head(mx.fast.rms_norm(hidden, self.norm, self.args.rms_norm_eps))

    def __call__(self, inputs, cache=None):
        hidden, _ = self.forward_hidden(inputs, cache)
        return self.project_hidden(hidden)


def load_model(path: str | Path, expert_store, progress: Callable | None = None, optimize=True,
               indexed_prefill=True, decode_k4=False) -> Model:
    """Load the 78 main layers; the optional MTP layer is not part of the backbone."""
    from .dense import DenseStore

    path = Path(path)
    args = ModelArgs.from_dict(json.loads((path / "config.json").read_text()))
    decode_k4 = (decode_k4 and optimize and getattr(expert_store, "optimize", False)
                 and getattr(expert_store, "resident", False))
    dense = DenseStore(path, optimize=optimize, decode_k4=decode_k4)
    layers = []
    for index in range(args.num_hidden_layers):
        weights = dense.layer(index)
        experts = expert_store.layer(index) if args.mlp_layer_types[index] == "sparse" else None
        layer = DecoderLayer(args, index, weights, experts)
        if decode_k4:
            from .kv_lookup import install

            install(layer.self_attn, weights["self_attn.kv_b_proj"])
            layer._fused_rms = (args.hidden_size == 6144 and args.rms_norm_eps == 1e-5
                                and layer.post_norm.shape == (6144,) and layer.post_norm.dtype == mx.bfloat16
                                and mx.device_info()["device_name"] == "Apple M3 Ultra")
        if optimize and indexed_prefill:
            from .dsa_indexed import install_indexed_prefill

            install_indexed_prefill(layer.self_attn)
        mx.eval(layer.self_attn.wk, layer.self_attn.wv)
        layers.append(layer)
        if progress is not None:
            progress(index + 1, args.num_hidden_layers)
    model = Model(args, layers, dense.vocab())
    mx.eval(model.parameters())
    return model
