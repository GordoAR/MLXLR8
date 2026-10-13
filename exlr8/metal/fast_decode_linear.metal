// Adapted from PonyExl3 ponyexl3/mlx/gemv_metal.py:_gemv_simd_source.
// Copyright 2026 Theinruj Toranavikrai. Apache-2.0.
// Modified for EXLR8 grouped128 storage, inline exact E4M3 decode, and
// selected expert rows. See NOTICE and third_party/licenses/.
#define PACKED_U32 (8u * K_BITS)

uint tn = threadgroup_position_in_grid.x;
uint row = threadgroup_position_in_grid.y;
uint split = threadgroup_position_in_grid.z;
uint tid = thread_position_in_threadgroup.x;
uint lane = tid & 31u;
uint sg = tid >> 5u;
uint in_tiles = dims[0];
uint out_tiles = dims[1];
uint splits = dims[3];
uint in_features = in_tiles * 16u;
uint out_features = out_tiles * 16u;
#if GATE_UP
uint projection = row / dims[2];
uint expert = indices[row % dims[2]];
const device uint* weights = projection ? packed_up : packed_gate;
#else
uint expert = indices[row];
const device uint* weights = packed;
#endif
#if HOT_CACHE
uint cache_slot = expert_to_slot[expert];
#if GATE_UP
const device uint* fp8_weights = projection ? cache_up : cache_gate;
#else
const device uint* fp8_weights = cache;
#endif
ulong source_base = ulong(expert) * in_tiles * out_tiles;
ulong cache_base = ulong(cache_slot) * in_tiles * out_tiles;
#endif
#if REUSE_ROWS > 1u
#if REUSE_ROWS == 2u
using row_value = float2;
using row_index = uint2;
#elif REUSE_ROWS == 3u
using row_value = float3;
using row_index = uint3;
#else
using row_value = float4;
using row_index = uint4;
#endif
uint first = row % dims[2];
uint repeats = 0u;
for (uint r = 0u; r < first; r++) repeats += uint(indices[r] == expert);
if (repeats % REUSE_ROWS != 0u) return;
row_index row_indices = row_index(0u);
uint row_count = 0u;
for (uint r = first; r < dims[2] && row_count < REUSE_ROWS; r++) {
    if (indices[r] == expert) {
#if GATE_UP
        row_indices[row_count++] = r + projection * dims[2];
#else
        row_indices[row_count++] = r;
#endif
    }
}
#else
using row_value = float;
#endif
uint span = (in_tiles + splits - 1u) / splits;
uint begin = split * span;
uint end = min(begin + span, in_tiles);

uint positions[8];
uint input_rows[8];
for (uint j = 0u; j < 8u; j++) {
    uint r = (lane & 3u) * 2u + (j & 1u) + (j & 2u) * 4u;
    uint c = (lane >> 2u) + (j & 4u) * 2u;
    positions[j] = r * 16u + c;
    input_rows[j] = r;
}
#if K_BITS != 4u || !DIRECT_K4
uint left[2], right[2], shifts[2];
for (uint g = 0u; g < 2u; g++) {
    uint last_bit = (lane * 8u + g * 4u + 4u) * K_BITS + 256u * K_BITS;
    uint last_word = (last_bit - 1u) / 32u;
    left[g] = (last_word - 1u) % PACKED_U32;
    right[g] = last_word % PACKED_U32;
    shifts[g] = (last_word + 1u) * 32u - last_bit;
}
#endif
row_value accum[8] = {row_value(0.0f), row_value(0.0f), row_value(0.0f), row_value(0.0f),
                      row_value(0.0f), row_value(0.0f), row_value(0.0f), row_value(0.0f)};
for (uint tk = begin + sg; tk < end; tk += NSG) {
    ulong tile;
#if OUTPUT_GROUPED
    tile = ((ulong(expert) * (out_tiles / 8u) + tn / 8u) * in_tiles + tk) * 8u + tn % 8u;
#else
    tile = (((ulong(expert) * (in_tiles / 8u) + tk / 8u) * (out_tiles / 8u) + tn / 8u) * 8u + tk % 8u) * 8u + tn % 8u;
#endif
    const device uint* words = weights + tile * PACKED_U32;
#if !DIRECT_X
#if REUSE_ROWS > 1u
    row_value x_lane = row_value(0.0f);
    #pragma unroll
    for (uint r = 0u; r < REUSE_ROWS; r++) {
        if (r < row_count) x_lane[r] = xh[ulong(row_indices[r]) * in_features + tk * 16u + (lane & 15u)];
    }
#else
    float x_lane = xh[ulong(row) * in_features + tk * 16u + (lane & 15u)];
#endif
#endif
#if HOT_CACHE
    if (cache_slot != 0xFFFFFFFFu) {
        const device uint* fp8_words = fp8_weights + (tile - source_base + cache_base) * 64u + lane * 2u;
        uint2 bytes = uint2(fp8_words[0], fp8_words[1]);
        float weight[8];
        #pragma unroll
        for (uint j = 0u; j < 4u; j++) {
            weight[j] = exlr8_fp8_byte((bytes.x >> (j * 8u)) & 255u);
            weight[j + 4u] = exlr8_fp8_byte((bytes.y >> (j * 8u)) & 255u);
        }
#if DIRECT_X
#if REUSE_ROWS > 1u
        exlr8_direct_fma<row_value, row_index, REUSE_ROWS>(accum, weight, xh, row_indices, row_count, in_features, tk, lane);
#else
        exlr8_direct_fma(accum, weight, xh, row, in_features, tk, lane);
#endif
#elif REUSE_ROWS > 1u
        #pragma unroll
        for (uint r = 0u; r < REUSE_ROWS; r++) {
            if (r < row_count) {
                #pragma unroll
                for (uint j = 0u; j < 4u; j++) {
                    float activation = simd_shuffle(x_lane[r], ushort(input_rows[j]));
                    accum[j][r] = fma(activation, weight[j], accum[j][r]);
                    accum[j + 4u][r] = fma(activation, weight[j + 4u], accum[j + 4u][r]);
                }
            }
        }
#else
        #pragma unroll
        for (uint j = 0u; j < 4u; j++) {
            float activation = simd_shuffle(x_lane, ushort(input_rows[j]));
            accum[j] = fma(activation, weight[j], accum[j]);
            accum[j + 4u] = fma(activation, weight[j + 4u], accum[j + 4u]);
        }
#endif
        continue;
    }
#endif
#if K_BITS == 4u && DIRECT_K4
    uint word = words[lane];
    uint prior = simd_shuffle(word, ushort((lane + 31u) & 31u));
    uint cw[8] = {((prior << 4u) | (word >> 28u)) & 0xFFFFu,
                  ((prior << 8u) | (word >> 24u)) & 0xFFFFu,
                  ((prior << 12u) | (word >> 20u)) & 0xFFFFu,
                  word >> 16u, (word >> 12u) & 0xFFFFu,
                  (word >> 8u) & 0xFFFFu, (word >> 4u) & 0xFFFFu,
                  word & 0xFFFFu};
    float weight[8];
    #pragma unroll
    for (uint j = 0u; j < 8u; j += 2u) {
#if LOOKUP
        weight[j] = float(codebook[cw[j]]);
        weight[j + 1u] = float(codebook[cw[j + 1u]]);
#elif VECTOR_DECODE
        float2 pair = exlr8_decode2(uint2(cw[j], cw[j + 1u]));
        weight[j] = pair.x;
        weight[j + 1u] = pair.y;
#else
        weight[j] = exlr8_decode(cw[j]);
        weight[j + 1u] = exlr8_decode(cw[j + 1u]);
#endif
    }
#if DIRECT_X
#if REUSE_ROWS > 1u
    exlr8_direct_fma<row_value, row_index, REUSE_ROWS>(accum, weight, xh, row_indices, row_count, in_features, tk, lane);
#else
    exlr8_direct_fma(accum, weight, xh, row, in_features, tk, lane);
#endif
#elif REUSE_ROWS > 1u
    #pragma unroll
    for (uint r = 0u; r < REUSE_ROWS; r++) {
        if (r < row_count) {
            #pragma unroll
            for (uint j = 0u; j < 4u; j++) {
                float activation = simd_shuffle(x_lane[r], ushort(input_rows[j]));
                accum[j][r] = fma(activation, weight[j], accum[j][r]);
                accum[j + 4u][r] = fma(activation, weight[j + 4u], accum[j + 4u][r]);
            }
        }
    }
#else
    #pragma unroll
    for (uint j = 0u; j < 4u; j++) {
        float activation = simd_shuffle(x_lane, ushort(input_rows[j]));
        accum[j] = fma(activation, weight[j], accum[j]);
        accum[j + 4u] = fma(activation, weight[j + 4u], accum[j + 4u]);
    }
#endif
#else
    for (uint g = 0u; g < 2u; g++) {
        ulong merged = (ulong(words[left[g]]) << 32u) | ulong(words[right[g]]);
        for (uint q = 0u; q < 4u; q++) {
            uint j = g * 4u + q;
            uint cw = uint(merged >> (shifts[g] + (3u - q) * K_BITS)) & 0xFFFFu;
#if LOOKUP
            float weight = float(codebook[cw]);
#else
            float weight = exlr8_decode(cw);
#endif
#if REUSE_ROWS > 1u
            #pragma unroll
            for (uint r = 0u; r < REUSE_ROWS; r++) {
                if (r < row_count) accum[j][r] = fma(simd_shuffle(x_lane[r], ushort(input_rows[j])), weight, accum[j][r]);
            }
#else
            accum[j] = fma(simd_shuffle(x_lane, ushort(input_rows[j])), weight, accum[j]);
#endif
        }
    }
#endif
}
#if QUAD_REDUCE
row_value lo = accum[0] + accum[1] + accum[2] + accum[3];
row_value hi = accum[4] + accum[5] + accum[6] + accum[7];
lo += simd_shuffle_xor(lo, ushort(1u));
hi += simd_shuffle_xor(hi, ushort(1u));
lo += simd_shuffle_xor(lo, ushort(2u));
hi += simd_shuffle_xor(hi, ushort(2u));
threadgroup row_value partial[NSG][16];
if ((lane & 3u) == 0u) {
    partial[sg][lane >> 2u] = lo;
    partial[sg][(lane >> 2u) + 8u] = hi;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 16u) {
    row_value value = row_value(0.0f);
    for (uint s = 0u; s < NSG; s++) value += partial[s][tid];
#if REUSE_ROWS > 1u
    #pragma unroll
    for (uint r = 0u; r < REUSE_ROWS; r++) {
        if (r < row_count) out[(ulong(row_indices[r]) * splits + split) * out_features + tn * 16u + tid] = value[r];
    }
#else
    out[(ulong(row) * splits + split) * out_features + tn * 16u + tid] = value;
#endif
}
#else
threadgroup row_value partial[NSG][256];
for (uint j = 0u; j < 8u; j++) partial[sg][positions[j]] = accum[j];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 16u) {
    row_value value = row_value(0.0f);
    for (uint r = 0u; r < 16u; r++) {
        uint p = r * 16u + tid;
        for (uint s = 0u; s < NSG; s++) value += partial[s][p];
    }
#if REUSE_ROWS > 1u
    #pragma unroll
    for (uint r = 0u; r < REUSE_ROWS; r++) {
        if (r < row_count) out[(ulong(row_indices[r]) * splits + split) * out_features + tn * 16u + tid] = value[r];
    }
#else
    out[(ulong(row) * splits + split) * out_features + tn * 16u + tid] = value;
#endif
}
#endif
