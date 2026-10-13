// Copyright (c) 2024 Apple Inc. SPDX-License-Identifier: MIT
// Adapted from MLX v0.32.3 rms_norm.metal, rms_looped.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
constexpr int N_READS = 4;
uint gid = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lsize = threads_per_threadgroup.x;
uint simd_lane_id = thread_index_in_simdgroup;
uint simd_group_id = simdgroup_index_in_threadgroup;
uint axis_size = 6144u;
uint w_stride = 1u;
float eps = epsilon[0];
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];
  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  residual += gid * size_t(axis_size) + lid * N_READS;
  residual_out += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        T rounded = T(x[i + r] + residual[i + r]);
        residual_out[i + r] = rounded;
        float xi = rounded;
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          T rounded = T(x[i + r] + residual[i + r]);
        residual_out[i + r] = rounded;
        float xi = rounded;
          acc += xi * xi;
        }
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[r + i] = w[w_stride * (i + r)] *
            static_cast<T>(T(x[r + i] + residual[r + i]) * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          out[r + i] = w[w_stride * (i + r)] *
              static_cast<T>(T(x[r + i] + residual[r + i]) * local_inv_mean[0]);
        }
      }
    }
  }
