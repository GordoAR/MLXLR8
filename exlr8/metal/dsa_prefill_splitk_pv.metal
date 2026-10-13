// Batched 32x32x16 F32 matrix tiles, four fixed K partitions.
// P and V stay BF16 in memory. Each V tile serves all rows in its threadgroup.
const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32u;
const uint lane = tid % 32u;
const uint row0 = threadgroup_position_in_grid.y * 32u;
const uint col0 = threadgroup_position_in_grid.x * 32u;
const uint batch = threadgroup_position_in_grid.z / 4u;
const uint part = threadgroup_position_in_grid.z % 4u;
threadgroup float pa[32 * 24];
threadgroup float va[16 * 40];
simdgroup_matrix<float, 8, 8> accum[2][2];
for (uint r = 0; r < 2; ++r)
    for (uint c = 0; c < 2; ++c)
        accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
for (uint kb = part * (K / 4); kb < (part + 1) * (K / 4); kb += 16) {
    for (uint i = tid; i < 32 * 16; i += 128) {
        const uint r = i / 16, d = i % 16;
        pa[r * 24 + d] = row0 + r < M ? float(p[(batch * M + row0 + r) * K + kb + d]) : 0.0f;
    }
    for (uint i = tid; i < 16 * 32; i += 128) {
        const uint d = i / 32, c = i % 32;
        va[d * 40 + c] = col0 + c < N ? float(v[(batch * K + kb + d) * N + col0 + c]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 0; d < 16; d += 8) {
        simdgroup_matrix<float, 8, 8> a[2], b[2];
        for (uint r = 0; r < 2; ++r)
            simdgroup_load(a[r], pa + ((sg / 2) * 16 + r * 8) * 24 + d, 24);
        for (uint c = 0; c < 2; ++c)
            simdgroup_load(b[c], va + d * 40 + (sg % 2) * 16 + c * 8, 40);
        for (uint r = 0; r < 2; ++r)
            for (uint c = 0; c < 2; ++c)
                simdgroup_multiply_accumulate(accum[r][c], a[r], b[c], accum[r][c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint r = 0; r < 2; ++r) {
    const uint row = row0 + (sg / 2) * 16 + r * 8 + ((lane >> 2) & 4) + ((lane >> 1) & 3);
    for (uint c = 0; c < 2; ++c) {
        const uint col = col0 + (sg % 2) * 16 + c * 8 + (((lane >> 2) & 2) | (lane & 1)) * 2;
        for (uint e = 0; e < 2; ++e)
            if (row < M && col + e < N)
                partial[((batch * 4 + part) * M + row) * N + col + e] = accum[r][c].thread_elements()[e];
    }
}
