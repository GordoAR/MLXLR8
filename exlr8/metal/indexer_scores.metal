// Adapted from MLX v0.32.3 Steel GEMM fragment order and tile loaders.
// https://github.com/ml-explore/mlx/tree/v0.32.3/mlx/backend/metal/kernels/steel/gemm
// Copyright © 2024 Apple Inc.
/*
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32u, lane = tid % 32u;
const uint query = threadgroup_position_in_grid.z;
const uint keys = dims[0];
const uint col0 = threadgroup_position_in_grid.x * BN;
const uint mr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
const uint nc = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
constexpr uint WM = 2u, WN = 2u, BK = 32u, TM = 2u, TN = BN / 16u;
constexpr uint A_READS = 8u, B_READS = BN / 4u;
threadgroup T qa[32 * 40];
threadgroup T ka[BN * 40];
threadgroup float weighted[32 * BN];
simdgroup_matrix<float, 8, 8> accum[TM][TN];
#pragma clang loop unroll(full)
for (uint r = 0; r < TM; ++r)
    for (uint c = 0; c < TN; ++c)
        accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
for (uint kb = 0; kb < 128u; kb += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ar = tid / (BK / A_READS), ad = (tid % (BK / A_READS)) * A_READS;
    const device T* qs = q + (ar * uint(ROWS) + query) * 128u + kb + ad;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < A_READS; j += 4u)
        *reinterpret_cast<threadgroup ushort4*>(qa + ar * 40u + ad + j) =
            *reinterpret_cast<const device ushort4*>(qs + j);
    const uint br = tid / (BK / B_READS), bd = (tid % (BK / B_READS)) * B_READS;
    const uint key = col0 + br;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < B_READS; j += 4u)
        *reinterpret_cast<threadgroup ushort4*>(ka + br * 40u + bd + j) = key < keys
            ? *reinterpret_cast<const device ushort4*>(k + key * 128u + kb + bd + j)
            : ushort4(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint d = 0; d < BK; d += 8u) {
        simdgroup_matrix<float, 8, 8> a[TM], b[TN];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; ++r) {
            const uint offset = (8u * (sg / WN) + mr + r * 8u * WM) * 40u + d + nc;
            a[r].thread_elements()[0] = float(qa[offset]);
            a[r].thread_elements()[1] = float(qa[offset + 1u]);
        }
        #pragma clang loop unroll(full)
        for (uint c = 0; c < TN; ++c) {
            const uint offset = (8u * (sg % WN) + nc + c * 8u * WN) * 40u + d + mr;
            b[c].thread_elements()[0] = float(ka[offset]);
            b[c].thread_elements()[1] = float(ka[offset + 40u]);
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; ++r)
            for (uint c = 0; c < TN; ++c)
                simdgroup_multiply_accumulate(accum[r][c], a[r], b[c], accum[r][c]);
    }
}
#pragma clang loop unroll(full)
for (uint r = 0; r < TM; ++r) {
    const uint head = 8u * (sg / WN) + mr + r * 8u * WM;
    const float weight = weights[head * uint(ROWS) + query];
    #pragma clang loop unroll(full)
    for (uint c = 0; c < TN; ++c) {
        const uint col = 8u * (sg % WN) + nc + c * 8u * WN;
        for (uint j = 0; j < 2u; ++j)
            weighted[(col + j) * 32u + head] =
                max(accum[r][c].thread_elements()[j] * 0.08838834764831845f, 0.0f) * weight;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint column = sg; column < BN; column += 4u) {
    const float sum = simd_sum(weighted[column * 32u + lane]);
    if (lane == 0u && col0 + column < keys) {
        const uint offset = query * keys + col0 + column;
        out[offset] = valid[offset] ? sum : -INFINITY;
    }
}
