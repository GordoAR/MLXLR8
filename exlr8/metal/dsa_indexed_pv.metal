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
// BF16 probabilities and values; F32 ordered products; final BF16 cast.
#define BN 64u
#define WM 1u
#define WN 2u
const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32u, lane = tid % 32u;
const uint query = threadgroup_position_in_grid.z;
const uint row0 = threadgroup_position_in_grid.y * 32u;
const uint col0 = threadgroup_position_in_grid.x * BN;
constexpr uint THREADS = 32u * WM * WN;
constexpr uint TM = 32u / (8u * WM), TN = BN / (8u * WN);
constexpr uint P_READS = 32u * 16u / THREADS;
constexpr uint V_READS = 16u * BN / THREADS;
const uint mr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
const uint nc = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
threadgroup T pa[32 * 24];
threadgroup T va[16 * (BN + 8)];
simdgroup_matrix<float, 8, 8> accum[TM][TN];
#pragma clang loop unroll(full)
for (uint r = 0; r < TM; ++r)
    for (uint c = 0; c < TN; ++c)
        accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
for (uint kb = 0; kb < 2048u; kb += 16u) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pr = tid / (16u / P_READS), pd = (tid % (16u / P_READS)) * P_READS;
    const device T* ps = p + (query * 64u + row0 + pr) * 2048u + kb + pd;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < P_READS; j += 4u)
        *reinterpret_cast<threadgroup ushort4*>(pa + pr * 24u + pd + j) =
            *reinterpret_cast<const device ushort4*>(ps + j);
    const uint vd = tid / (BN / V_READS), vc = (tid % (BN / V_READS)) * V_READS;
    const uint key = ids[query * 2048u + kb + vd];
    const device T* vs = v + key * 512u + col0 + vc;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < V_READS; j += 4u)
        *reinterpret_cast<threadgroup ushort4*>(va + vd * (BN + 8u) + vc + j) =
            *reinterpret_cast<const device ushort4*>(vs + j);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 16u; d += 8u) {
        simdgroup_matrix<float, 8, 8> a[TM], b[TN];
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; ++r) {
            const uint offset = (8u * (sg / WN) + mr + r * 8u * WM) * 24u + d + nc;
            a[r].thread_elements()[0] = float(pa[offset]);
            a[r].thread_elements()[1] = float(pa[offset + 1u]);
        }
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (uint c = 0; c < TN; ++c) {
            const uint offset = (d + mr) * (BN + 8u) + 8u * (sg % WN) + nc + c * 8u * WN;
            b[c].thread_elements()[0] = float(va[offset]);
            b[c].thread_elements()[1] = float(va[offset + 1u]);
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
    const uint row = row0 + 8u * (sg / WN) + mr + r * 8u * WM;
    #pragma clang loop unroll(full)
    for (uint c = 0; c < TN; ++c) {
        const uint col = col0 + 8u * (sg % WN) + nc + c * 8u * WN;
        const uint offset = (query * 64u + row) * 512u + col;
        out[offset] = T(accum[r][c].thread_elements()[0]);
        out[offset + 1u] = T(accum[r][c].thread_elements()[1]);
    }
}
