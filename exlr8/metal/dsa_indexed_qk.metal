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
// BF16 tile storage; F32 ordered 8-wide products; final BF16 cast.
#define BN 64u
#define BK 16u
#define WM 1u
#define WN 2u
const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32u, lane = tid % 32u;
const uint query = threadgroup_position_in_grid.z;
const uint col0 = threadgroup_position_in_grid.x * BN;
constexpr uint THREADS = 32u * WM * WN;
constexpr uint TM = 64u / (8u * WM), TN = BN / (8u * WN);
constexpr uint A_READS = 64u * BK / THREADS, B_READS = BN * BK / THREADS;
const uint mr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
const uint nc = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
threadgroup T qa[64 * (BK + 8)];
threadgroup T ka[BN * (BK + 8)];
simdgroup_matrix<float, 8, 8> accum[TM][TN];
#pragma clang loop unroll(full)
for (uint r = 0; r < TM; ++r)
    for (uint c = 0; c < TN; ++c)
        accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
for (uint kb = 0; kb < 512u; kb += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ar = tid / (BK / A_READS), ad = (tid % (BK / A_READS)) * A_READS;
    const device T* qs = q + (query * 64u + ar) * 512u + kb + ad;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < A_READS; j += 4u)
        *reinterpret_cast<threadgroup ushort4*>(qa + ar * (BK + 8u) + ad + j) =
            *reinterpret_cast<const device ushort4*>(qs + j);
    const uint br = tid / (BK / B_READS), bd = (tid % (BK / B_READS)) * B_READS;
    const uint key = ids[query * 2048u + col0 + br];
    const device T* ks = latent + key * 512u + kb + bd;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < B_READS; j += 4u)
        *reinterpret_cast<threadgroup ushort4*>(ka + br * (BK + 8u) + bd + j) =
            *reinterpret_cast<const device ushort4*>(ks + j);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint d = 0; d < BK; d += 8u) {
        simdgroup_matrix<float, 8, 8> a[TM], b[TN];
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; ++r) {
            const uint offset = (8u * (sg / WN) + mr + r * 8u * WM) * (BK + 8u) + d + nc;
            a[r].thread_elements()[0] = float(qa[offset]);
            a[r].thread_elements()[1] = float(qa[offset + 1u]);
        }
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (uint c = 0; c < TN; ++c) {
            const uint offset = (8u * (sg % WN) + nc + c * 8u * WN) * (BK + 8u) + d + mr;
            b[c].thread_elements()[0] = float(ka[offset]);
            b[c].thread_elements()[1] = float(ka[offset + BK + 8u]);
        }
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; ++r) {
            #pragma clang loop unroll(full)
            for (uint c = 0; c < TN; ++c) {
                const uint column = (r % 2u) ? TN - 1u - c : c;
                simdgroup_multiply_accumulate(accum[r][column], a[r], b[column], accum[r][column]);
            }
        }
    }
}
#pragma clang loop unroll(full)
for (uint r = 0; r < TM; ++r) {
    const uint row = 8u * (sg / WN) + mr + r * 8u * WM;
    #pragma clang loop unroll(full)
    for (uint c = 0; c < TN; ++c) {
        const uint col = col0 + 8u * (sg % WN) + nc + c * 8u * WN;
        const uint offset = (query * 64u + row) * 2048u + col;
        out[offset] = T(accum[r][c].thread_elements()[0]);
        out[offset + 1u] = T(accum[r][c].thread_elements()[1]);
    }
}
