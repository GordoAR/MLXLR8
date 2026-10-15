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
// Four query rows share exact BF16 weights and the native GEMVT sum order.
uint head = threadgroup_position_in_grid.z;
uint lane = thread_index_in_simdgroup;
uint first = threadgroup_position_in_grid.x * 64u
    + (simdgroup_index_in_threadgroup * 4u + lane % 4u) * 4u;
uint input_start = (lane / 4u) * 4u;
float4 result[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
for (uint block = 0; block < 192u; block += 32u) {
    threadgroup_barrier(mem_flags::mem_none);
    #pragma unroll
    for (uint j = 0; j < 4u; ++j) {
        uint input_col = block + input_start + j;
        float4 activation;
        #pragma unroll
        for (uint row = 0; row < 4u; ++row)
            activation[row] = float(x[(row * 64u + head) * 256u + input_col]);
        uint table_base = (((head * 448u + input_col) / 128u) * 4u + first / 128u) * 256u;
        uchar4 codes = *reinterpret_cast<const device uchar4*>(weights +
            (head * 192u + input_col) * 512u + first);
        #pragma unroll
        for (uint col = 0; col < 4u; ++col)
            result[col] += activation * float(table[table_base + codes[col]]);
    }
}
#pragma unroll
for (uint col = 0; col < 4u; ++col) {
    #pragma unroll
    for (ushort stride = 16; stride >= 4; stride >>= 1)
        result[col] += simd_shuffle_down(result[col], stride);
    if (lane / 4u == 0)
        #pragma unroll
        for (uint row = 0; row < 4u; ++row)
            out[(head * 4u + row) * 512u + first + col] = bfloat16_t(result[col][row]);
}
