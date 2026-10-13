// Mixed half operands / float accumulators follow fast_prefill.metal's MMA
// calls, adapted from PonyExl3 (Apache-2.0), Copyright 2026 Theinruj
// Toranavikrai. See NOTICE and third_party/licenses/.
// This kernel loads unquantized BF16/F32 tiles and writes the input type.
#define A_STRIDE 40u
#define B_STRIDE 72u

uint m = dims[0], n = dims[1], k = dims[2];
uint row_block = threadgroup_position_in_grid.y * 64u;
uint col_block = threadgroup_position_in_grid.x * 64u;
uint batch = threadgroup_position_in_grid.z;
uint tid = thread_position_in_threadgroup.x;
uint lane = tid & 31u, sg = tid >> 5u;
uint sg_row = sg / 2u, sg_col = sg % 2u;

threadgroup half aa[64u * A_STRIDE];
threadgroup half bb[32u * B_STRIDE];
simdgroup_float8x8 accum[4][4];
for (uint r = 0u; r < 4u; r++) {
    for (uint c = 0u; c < 4u; c++) {
        accum[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
}

for (uint stage = 0u; stage < k; stage += 32u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 64u * 32u; i += 128u) {
        uint row = row_block + i / 32u, inner = stage + i % 32u;
        aa[(i / 32u) * A_STRIDE + i % 32u] = row < m && inner < k
            ? half(a[(ulong(batch) * m + row) * k + inner]) : half(0.0f);
    }
    for (uint i = tid; i < 32u * 64u; i += 128u) {
        uint inner = stage + i / 64u, col = col_block + i % 64u;
#if TRANSPOSE_B
        ulong source = (ulong(batch) * n + col) * k + inner;
#else
        ulong source = (ulong(batch) * k + inner) * n + col;
#endif
        bb[(i / 64u) * B_STRIDE + i % 64u] = inner < k && col < n
            ? half(b[source]) : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0u; inner < 32u; inner += 8u) {
        simdgroup_half8x8 a_frag[4], b_frag[4];
        for (uint r = 0u; r < 4u; r++) {
            simdgroup_load(a_frag[r], aa + (sg_row * 32u + r * 8u) * A_STRIDE + inner, A_STRIDE);
        }
        for (uint c = 0u; c < 4u; c++) {
            simdgroup_load(b_frag[c], bb + inner * B_STRIDE + sg_col * 32u + c * 8u, B_STRIDE);
        }
        for (uint r = 0u; r < 4u; r++) {
            for (uint c = 0u; c < 4u; c++) {
                simdgroup_multiply_accumulate(accum[r][c], a_frag[r], b_frag[c], accum[r][c]);
            }
        }
    }
}

for (uint r = 0u; r < 4u; r++) {
    for (uint c = 0u; c < 4u; c++) {
        // Fragment coordinates follow MLX Steel (MIT), Copyright Apple Inc.
        uint row = row_block + sg_row * 32u + r * 8u + ((lane >> 2u) & 4u) + ((lane >> 1u) & 3u);
        uint col = col_block + sg_col * 32u + c * 8u + (((lane >> 2u) & 2u) | (lane & 1u)) * 2u;
        thread float2& elements = reinterpret_cast<thread float2&>(accum[r][c].thread_elements());
        if (row < m && col < n) {
            out[(ulong(batch) * m + row) * n + col] = U(elements[0]);
        }
        if (row < m && col + 1u < n) {
            out[(ulong(batch) * m + row) * n + col + 1u] = U(elements[1]);
        }
    }
}
