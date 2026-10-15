// Exact expert finish, routed sum, shared sum, and residual sum.
// Sum order follows MLX v0.32.3 col_reduce_small:
// mlx/backend/metal/kernels/reduction/reduce_col.h.
// Copyright (c) 2023-2024 Apple Inc. MIT License.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Nine SIMD groups work on the nine selected experts at the same time.
uint block = threadgroup_position_in_grid.x;
uint token = threadgroup_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x & 31u;
uint slot = thread_position_in_threadgroup.x >> 5u;
uint features = dims[0];
uint row = token * 9u + slot;
uint inner_row = permutation[row];
uint expert = indices[row];
float4 value = float4(0.0f);
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    value[j] += inner[ulong(inner_row) * features + col];
}
value = exlr8_hadamard128(value, lane);
threadgroup float products[9u * 128u];
for (uint j = 0u; j < 4u; j++) {
    uint local_col = lane + j * 32u;
    uint col = block * 128u + local_col;
    T expert_value = T(value[j] * float(out_scale[ulong(expert) * features + col]));
    products[slot * 128u + local_col] = slot < 8u
        ? float(expert_value) * scores[token * 8u + slot]
        : float(expert_value);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (slot == 0u) {
    for (uint j = 0u; j < 4u; j++) {
        uint local_col = lane + j * 32u;
        uint col = block * 128u + local_col;
        // MLX 0.32.3 col_reduce_small uses eight row threads. Each
        // starts at +0, then row 0 adds rows 1 through 7 in this order.
        float total = products[local_col] + 0.0f;
        for (uint r = 1u; r < 8u; r++) {
            total = (products[r * 128u + local_col] + 0.0f) + total;
        }
        T routed = T(total);
        T combined = T(float(routed) + float(T(products[8u * 128u + local_col])));
        ulong offset = ulong(token) * features + col;
        out[offset] = T(float(residual[offset]) + float(combined));
    }
}
