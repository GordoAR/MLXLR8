// Adapted from PonyExl3 ponyexl3/mlx/gemv_metal.py:_mm_seg_source.
// Copyright 2026 Theinruj Toranavikrai. Licensed under Apache-2.0.
// Modified for EXLR8: grouped128 storage, exact inline E4M3 decoding, runtime
// expert dimensions, smaller row blocks, and float32 accumulation.
// Experimental FP16 operands preserve the exact E4M3 weight values.
// Bitstream geometry and tile order: exllamav3 0.0.43 (MIT),
// Copyright (c) 2025 Turboderp. See NOTICE and third_party/licenses/.
#define PACKED_U32 (8u * K_BITS)
#define WEIGHT_STRIDE 72u
#if BLOCK_ROWS <= 16
#define SIMD_ROWS 1u
#define SIMD_COLS 4u
#else
#define SIMD_ROWS 2u
#define SIMD_COLS 2u
#endif
#define ROW_FRAGMENTS (BLOCK_ROWS / SIMD_ROWS / 8u)
#define COL_FRAGMENTS (64u / SIMD_COLS / 8u)
#if HALF_DOT
typedef half dot_type;
#define OPERAND_MATRIX simdgroup_half8x8
#else
typedef float dot_type;
#define OPERAND_MATRIX simdgroup_float8x8
#endif

uint block = threadgroup_position_in_grid.y;
if (block >= count[0]) {
    return;
}
uint col_block = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = tid & 31u;
uint sg = tid >> 5u;
uint sg_row = sg / SIMD_COLS;
uint sg_col = sg % SIMD_COLS;
uint in_tiles = dims[0];
uint out_tiles = dims[1];
uint blocks = dims[2];
uint in_features = in_tiles * 16u;
uint out_features = out_tiles * 16u;
uint expert = table[block];
uint first_row = table[blocks + block];
uint length = table[2u * blocks + block];
uint tn0 = col_block * 4u;
uint projection = threadgroup_position_in_grid.z;
#if GATE_UP
const device uint* matrix = projection == 0u ? packed_gate : packed_up;
#else
const device uint* matrix = packed;
#endif

#if K_BITS != 4 || BLOCK_ROWS > 32
uint tile_col = tid >> 3u;
uint tile_row_pair = tid & 7u;
uint left[2], right[2], shift[2];
for (uint j = 0u; j < 2u; j++) {
    uint r = 2u * tile_row_pair + j;
    uint code_lane = (tile_col & 7u) * 4u + ((r & 7u) >> 1u);
    uint slot = (r & 1u) + ((r & 8u) >> 2u) + ((tile_col & 8u) >> 1u);
    uint last_bit = (code_lane * 8u + slot + 1u) * K_BITS + 256u * K_BITS;
    uint word = (last_bit - 1u) / 32u;
    left[j] = (word - 1u) % PACKED_U32;
    right[j] = word % PACKED_U32;
    shift[j] = (word + 1u) * 32u - last_bit;
}
#endif

threadgroup dot_type weights[32u * WEIGHT_STRIDE];
threadgroup float scratch[4u * 64u];
simdgroup_float8x8 accum[ROW_FRAGMENTS][COL_FRAGMENTS];
for (uint r = 0u; r < ROW_FRAGMENTS; r++) {
    for (uint c = 0u; c < COL_FRAGMENTS; c++) {
        accum[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
}

uint sg_first = sg_row * ROW_FRAGMENTS * 8u;
const device dot_type* xrow = xh + (ulong(projection) * dims[4] + first_row + sg_first) * in_features;
for (uint stage = 0u; stage < in_tiles / 2u; stage++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint input_tile = 0u; input_tile < 2u; input_tile++) {
        uint tk = stage * 2u + input_tile;
#if K_BITS == 4 && BLOCK_ROWS <= 32
        uint tn = tn0 + sg;
        ulong tile;
#if OUTPUT_GROUPED
        tile = ((ulong(expert) * (out_tiles / 8u) + tn / 8u) * in_tiles + tk) * 8u + tn % 8u;
#else
        tile = (((ulong(expert) * (in_tiles / 8u) + tk / 8u) * (out_tiles / 8u) + tn / 8u) * 8u + tk % 8u) * 8u + tn % 8u;
#endif
        uint word = matrix[tile * 32u + lane];
        uint prior = simd_shuffle(word, ushort((lane + 31u) & 31u));
        uint cw[8] = {((prior << 4u) | (word >> 28u)) & 0xFFFFu,
                      ((prior << 8u) | (word >> 24u)) & 0xFFFFu,
                      ((prior << 12u) | (word >> 20u)) & 0xFFFFu,
                      word >> 16u, (word >> 12u) & 0xFFFFu,
                      (word >> 8u) & 0xFFFFu, (word >> 4u) & 0xFFFFu,
                      word & 0xFFFFu};
        #pragma unroll
        for (uint j = 0u; j < 8u; j += 2u) {
            float2 value = exlr8_decode2(uint2(cw[j], cw[j + 1u]));
            uint r = (lane & 3u) * 2u + (j & 2u) * 4u;
            uint c = (lane >> 2u) + (j & 4u) * 2u;
            weights[(input_tile * 16u + r) * WEIGHT_STRIDE + sg * 16u + c] = dot_type(value.x);
            weights[(input_tile * 16u + r + 1u) * WEIGHT_STRIDE + sg * 16u + c] = dot_type(value.y);
        }
#else
        for (uint output_tile = 0u; output_tile < 4u; output_tile++) {
            uint tn = tn0 + output_tile;
            ulong tile;
#if OUTPUT_GROUPED
            tile = ((ulong(expert) * (out_tiles / 8u) + tn / 8u) * in_tiles + tk) * 8u + tn % 8u;
#else
            tile = (((ulong(expert) * (in_tiles / 8u) + tk / 8u) * (out_tiles / 8u) + tn / 8u) * 8u + tk % 8u) * 8u + tn % 8u;
#endif
            const device uint* words = matrix + tile * PACKED_U32;
            for (uint j = 0u; j < 2u; j++) {
                ulong merged = (ulong(words[left[j]]) << 32u) | ulong(words[right[j]]);
                uint cw = uint(merged >> shift[j]) & 0xFFFFu;
                weights[(input_tile * 16u + 2u * tile_row_pair + j) * WEIGHT_STRIDE + output_tile * 16u + tile_col] = dot_type(exlr8_decode(cw));
            }
        }
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_first >= length) {
        continue;
    }
    for (uint ki = 0u; ki < 4u; ki++) {
        OPERAND_MATRIX b[COL_FRAGMENTS];
        const threadgroup dot_type* wrow = weights + ki * 8u * WEIGHT_STRIDE + sg_col * COL_FRAGMENTS * 8u;
        for (uint c = 0u; c < COL_FRAGMENTS; c++) {
            simdgroup_load(b[c], wrow + c * 8u, WEIGHT_STRIDE);
        }
        for (uint r = 0u; r < ROW_FRAGMENTS; r++) {
            if (sg_first + r * 8u >= length) {
                break;
            }
            OPERAND_MATRIX a;
            simdgroup_load(a, xrow + ulong(r * 8u) * in_features + stage * 32u + ki * 8u, in_features);
            for (uint c = 0u; c < COL_FRAGMENTS; c++) {
                simdgroup_multiply_accumulate(accum[r][c], a, b[c], accum[r][c]);
            }
        }
    }
}

uint first_col = col_block * 64u + sg_col * COL_FRAGMENTS * 8u;
for (uint r = 0u; r < ROW_FRAGMENTS; r++) {
    if (sg_first + r * 8u >= length) {
        break;
    }
    for (uint c = 0u; c < COL_FRAGMENTS; c++) {
        simdgroup_store(accum[r][c], scratch + sg * 64u, 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64u; i += 32u) {
            uint row = sg_first + r * 8u + (i >> 3u);
            if (row < length) {
                out[(ulong(projection) * dims[3] + first_row + row) * out_features + first_col + c * 8u + (i & 7u)] = scratch[sg * 64u + i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
