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
// Four rows share the exact BF16 lookup weights and native GEMV sum order.
uint head = threadgroup_position_in_grid.z;
uint lane = thread_index_in_simdgroup;
uint first = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u;
float4 result[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
for (uint block = 0; block < 512u; block += 128u) {
    uint column = block + lane * 4u;
    uint table_base = (((head * 448u + 192u + first) / 128u) * 4u + block / 128u) * 256u;
    float4 activation[4];
    #pragma unroll
    for (uint j = 0; j < 4u; ++j)
        #pragma unroll
        for (uint row = 0; row < 4u; ++row)
            activation[j][row] = float(x[(head * 4u + row) * 512u + column + j]);
    #pragma unroll
    for (uint out_col = 0; out_col < 4u; ++out_col) {
        uchar4 codes = *reinterpret_cast<const device uchar4*>(weights +
            (head * 256u + first + out_col) * 512u + column);
        #pragma unroll
        for (uint j = 0; j < 4u; ++j)
            result[out_col] += activation[j] * float(table[table_base + codes[j]]);
    }
}
#pragma unroll
for (uint out_col = 0; out_col < 4u; ++out_col) {
    #pragma unroll
    for (ushort stride = 16; stride >= 1; stride >>= 1)
        result[out_col] += simd_shuffle_down(result[out_col], stride);
    if (lane == 0)
        #pragma unroll
        for (uint row = 0; row < 4u; ++row)
            out[(row * 64u + head) * 256u + first + out_col] = bfloat16_t(result[out_col][row]);
}
