// Adapted from PonyExl3 ponyexl3/mlx/gemv_metal.py:_gemv_simd_source.
// Copyright 2026 Theinruj Toranavikrai. Licensed under Apache-2.0.
// Modified for EXLR8: direct grouped128 storage, compact expert indices,
// FP8-rounded codebook lookup, and one activation row per expert slot.
// Bitstream geometry and tile order: exllamav3 0.0.43 (MIT),
// Copyright (c) 2025 Turboderp. See NOTICE and third_party/licenses/.
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
uint expert = indices[row];
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
uint left[2], right[2], shifts[2];
for (uint g = 0u; g < 2u; g++) {
    uint last_bit = (lane * 8u + g * 4u + 4u) * K_BITS + 256u * K_BITS;
    uint last_word = (last_bit - 1u) / 32u;
    left[g] = (last_word - 1u) % PACKED_U32;
    right[g] = last_word % PACKED_U32;
    shifts[g] = (last_word + 1u) * 32u - last_bit;
}
float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
for (uint tk = begin + sg; tk < end; tk += 4u) {
    ulong tile;
#if OUTPUT_GROUPED
    tile = ((ulong(expert) * (out_tiles / 8u) + tn / 8u) * in_tiles + tk) * 8u + tn % 8u;
#else
    tile = (((ulong(expert) * (in_tiles / 8u) + tk / 8u) * (out_tiles / 8u) + tn / 8u) * 8u + tk % 8u) * 8u + tn % 8u;
#endif
    const device uint* words = packed + tile * PACKED_U32;
    float x_lane = xh[ulong(row) * in_features + tk * 16u + (lane & 15u)];
    for (uint g = 0u; g < 2u; g++) {
        ulong merged = (ulong(words[left[g]]) << 32u) | ulong(words[right[g]]);
        for (uint q = 0u; q < 4u; q++) {
            uint j = g * 4u + q;
            uint cw = uint(merged >> (shifts[g] + (3u - q) * K_BITS)) & 0xFFFFu;
            float weight = float(codebook[cw]);
            accum[j] = fma(simd_shuffle(x_lane, ushort(input_rows[j])), weight, accum[j]);
        }
    }
}
threadgroup float partial[4][256];
for (uint j = 0u; j < 8u; j++) {
    partial[sg][positions[j]] = accum[j];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 16u) {
    float value = 0.0f;
    for (uint r = 0u; r < 16u; r++) {
        uint p = r * 16u + tid;
        value += partial[0][p] + partial[1][p] + partial[2][p] + partial[3][p];
    }
    out[(ulong(row) * splits + split) * out_features + tn * 16u + tid] = value;
}
