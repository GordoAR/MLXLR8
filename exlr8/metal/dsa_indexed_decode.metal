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
// Q1 only. Keep each BF16 boundary and the four native F32 PV partitions.
#if STAGE == 2
const uint i = thread_position_in_grid.x;
float total = 0.0f;
for (uint part = 0; part < 4u; ++part)
    total += partial[part * 32768u + i];
out[i] = T(total);
#elif STAGE == 0
const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32u, lane = tid % 32u;
const uint mr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
const uint nc = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
constexpr uint WM = 2u, WN = 2u, BN = 32u, THREADS = 128u;
constexpr uint TM = 4u, TN = 2u;
const uint col0 = threadgroup_position_in_grid.x * BN;

// Native Q1 QK and PE both use NT BM64/BN32/BK32/WM2/WN2.
constexpr uint BK = 32u, A_READS = 16u, B_READS = 8u;
threadgroup T qa[64 * 40];
threadgroup T ka[32 * 40];
thread T nope[TM][TN][2];
#pragma clang loop unroll(full)
for (uint phase = 0; phase < 2u; ++phase) {
    simdgroup_matrix<float, 8, 8> accum[TM][TN];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; ++r)
        for (uint c = 0; c < TN; ++c)
            accum[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
    const uint dims = phase == 0u ? 512u : 64u;
    for (uint kb = 0; kb < dims; kb += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint ar = tid / (BK / A_READS), ad = (tid % (BK / A_READS)) * A_READS;
        const device T* qs = phase == 0u ? q + ar * 512u + kb + ad : qp + ar * 64u + kb + ad;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < A_READS; j += 4u)
            *reinterpret_cast<threadgroup ushort4*>(qa + ar * 40u + ad + j) =
                *reinterpret_cast<const device ushort4*>(qs + j);
        const uint br = tid / (BK / B_READS), bd = (tid % (BK / B_READS)) * B_READS;
        const uint key = ids[col0 + br];
        const device T* ks = phase == 0u ? latent + key * 512u + kb + bd : pe + key * 64u + kb + bd;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < B_READS; j += 4u)
            *reinterpret_cast<threadgroup ushort4*>(ka + br * 40u + bd + j) =
                *reinterpret_cast<const device ushort4*>(ks + j);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint d = 0; d < BK; d += 8u) {
            simdgroup_matrix<float, 8, 8> a[TM], b[TN];
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; ++r) {
                const uint offset = (8u * (sg / WN) + mr + r * 8u * WM) * 40u + d + nc;
                a[r].thread_elements()[0] = float(qa[offset]);
                a[r].thread_elements()[1] = float(qa[offset + 1u]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint c = 0; c < TN; ++c) {
                const uint offset = (8u * (sg % WN) + nc + c * 8u * WN) * 40u + d + mr;
                b[c].thread_elements()[0] = float(ka[offset]);
                b[c].thread_elements()[1] = float(ka[offset + 40u]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < TM; ++r)
                for (uint c = 0; c < TN; ++c) {
                    const uint column = (r % 2u) ? TN - 1u - c : c;
                    simdgroup_multiply_accumulate(accum[r][column], a[r], b[column], accum[r][column]);
                }
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; ++r) {
        const uint row = 8u * (sg / WN) + mr + r * 8u * WM;
        #pragma clang loop unroll(full)
        for (uint c = 0; c < TN; ++c) {
            const uint col = col0 + 8u * (sg % WN) + nc + c * 8u * WN;
            const uint offset = row * 2048u + col;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 2u; ++j) {
                const T value = T(accum[r][c].thread_elements()[j]);
                if (phase == 0u) {
                    nope[r][c][j] = value;
                } else {
                    const T bias = valid[col + j] ? value : T(-INFINITY);
                    out[offset + j] = T(float(nope[r][c][j]) + float(bias));
                }
            }
        }
    }
}
#else
#define BN 32u
#define WM 2u
#define WN 2u
#define PARTS 4u
const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32u, lane = tid % 32u;
const uint query = threadgroup_position_in_grid.z / PARTS;
const uint part = threadgroup_position_in_grid.z % PARTS;
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
for (uint kb = part * (2048u / PARTS); kb < (part + 1u) * (2048u / PARTS); kb += 16u) {
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
        const uint offset = ((query * PARTS + part) * 64u + row) * 512u + col;
        out[offset] = float(accum[r][c].thread_elements()[0]);
        out[offset + 1u] = float(accum[r][c].thread_elements()[1]);
    }
}
#endif
