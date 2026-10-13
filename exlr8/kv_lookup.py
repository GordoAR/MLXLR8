"""Byte lookup for the tested small-batch attention projections."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


RECORD = (4, 28672, 512, 0, 1, 448, 0, 1)


@lru_cache(maxsize=1)
def _table_kernel():
    source = (Path(__file__).with_name("metal") / "kv_lookup_table.metal").read_text()
    return mx.fast.metal_kernel(name="exlr8_kv_lookup_table", input_names=["scales"],
                               output_names=["table"], source=source)


def prepare(packed, scales):
    codes = packed.view(mx.uint8).reshape(64, 448, 512)
    table = _table_kernel()(inputs=[scales.reshape(-1)], grid=(896 * 256, 1, 1),
                           threadgroup=(256, 1, 1), output_shapes=[(896, 256)],
                           output_dtypes=[mx.bfloat16])[0]
    return mx.contiguous(codes[:, :192]), mx.contiguous(codes[:, 192:]), table


def install(attention, operator):
    from .dense import DenseLinear

    a = attention.args
    if (not isinstance(operator, DenseLinear) or not operator.optimize
            or operator.record != RECORD
            or (a.num_attention_heads, a.qk_nope_head_dim, a.qk_rope_head_dim,
                a.v_head_dim, a.kv_lora_rank) != (64, 192, 64, 256, 512)):
        return False
    attention.wk_codes, attention.wv_codes, attention.kv_lookup_table = prepare(
        operator.packed, operator.scales)
    mx.eval(attention.wk_codes, attention.wv_codes, attention.kv_lookup_table)
    attention._lookup = True
    return True


@lru_cache(maxsize=2)
def _kernel(query):
    header, source = (Path(__file__).with_name("metal") / "kv_lookup.metal").read_text().split(
        "// KERNEL_ENTRY\n", 1)
    header = (f"#define KEY_PROJECTION {int(query)}\n#define LOOKUP true\n"
              "#define WEIGHT_TYPE uint8_t\n"
              f"#define INPUT_STRIDE {256 if query else 512}\n"
              f"#define INPUT_SIZE {192 if query else 512}\n"
              f"#define OUTPUT_SIZE {512 if query else 256}\n"
              f"#define OUTPUT_ROWS {192 if query else 256}\n"
              f"#define ROW_OFFSET {0 if query else 192}\n") + header
    return mx.fast.metal_kernel(name=f"exlr8_kv_lookup_{'query' if query else 'value'}",
                               input_names=["x", "weights", "table"], output_names=["out"],
                               header=header, source=source)


@mx.compile
def query_projection(x, weights, table):
    rows = x.shape[2]
    return _kernel(True)(inputs=[x, weights, table], template=[("TOKENS_PER_HEAD", rows)],
                         grid=(128 * 8, rows, 64),
                         threadgroup=(128, 1, 1), output_shapes=[(1, 64, rows, 512)],
                         output_dtypes=[mx.bfloat16])[0]


@mx.compile
def value_projection(x, weights, table):
    rows = x.shape[2]
    return _kernel(False)(inputs=[x, weights, table], template=[("TOKENS_PER_HEAD", rows)],
                          grid=(128 * 16, rows, 64),
                          threadgroup=(128, 1, 1), output_shapes=[(1, 64, rows, 256)],
                          output_dtypes=[mx.bfloat16])[0]
