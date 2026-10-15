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
// E4M3 decoder adapted from MLX 0.32.3 fp8.h.
// One or four vocabulary rows share packed reads and apply the original F32 row scales.
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// BEGIN fp8_e4m3_decoder
struct fp8_e4m3 {
  operator float16_t() thread {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return converted * 256.0;
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  uint8_t bits;
};

inline float fp8_decode(uchar code) {
    return float(*(thread fp8_e4m3*)(&code));
}

// KERNEL_ENTRY
#if BATCH_ROWS == 1
using value = float;
#else
using value = float4;
#endif
uint lane = thread_index_in_simdgroup;
uint first = threadgroup_position_in_grid.y * 8u + simdgroup_index_in_threadgroup * 4u;
value result[4] = {value(0.0f), value(0.0f), value(0.0f), value(0.0f)};
for (uint block = 0u; block < INPUT_K; block += 256u) {
    value activation[8];
    uint input = block + lane * 8u;
    #pragma unroll
    for (short i = 0; i < 8; i++) {
#if BATCH_ROWS == 1
        activation[i] = xh[input + uint(i)];
#else
        #pragma unroll
        for (short row = 0; row < 4; row++)
            activation[i][row] = xh[uint(row) * INPUT_K + input + uint(i)];
#endif
    }
    #pragma unroll
    for (short column = 0; column < 4; column++) {
        value dot(0.0f);
        uint2 packed = reinterpret_cast<const device uint2*>(weights)[
            (first + uint(column)) * (INPUT_K / 8) + input / 8u];
        #pragma unroll
        for (short i = 0; i < 8; i++)
            dot += activation[i] * fp8_decode(uchar(packed[i / 4] >> (8 * (i % 4))));
        result[column] += dot;
    }
}
#pragma unroll
for (short column = 0; column < 4; column++) {
    value sum = simd_sum(result[column]);
    if (lane == 0u) {
#if BATCH_ROWS == 1
        out[first + uint(column)] = T(sum * row_scales[first + uint(column)]);
#else
        #pragma unroll
        for (short row = 0; row < 4; row++)
            out[uint(row) * OUTPUT_N + first + uint(column)] = T(sum[row] * row_scales[first + uint(column)]);
#endif
    }
}
