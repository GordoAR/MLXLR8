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
// Four rows share each byte load; the F32 reduction order is unchanged.
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

inline float fp8_fetch(const device uint* weights, uint c, uint r) {
    uchar code = reinterpret_cast<const device uchar*>(weights)[c * INPUT_K + r];
    return float(*(thread fp8_e4m3*)(&code));
}

inline float fp8_scale(uchar value) {
    uint bits = value == 0 ? 0x400000u : uint(value) << 23u;
    return as_type<float>(bits);
}

template <short ROWS>
METAL_FUNC void fp8_shared_rows(
    const device uint* weights, const device uchar* scales,
    const device float* x, device float* out,
    uint3 group, uint simd_group, uint lane) {
    using value = vec<float, ROWS>;
    constexpr int VALUES = 8;
    uint first = group.y * 8u + simd_group * 4u;
    value result[4] = {value(0.0f), value(0.0f), value(0.0f), value(0.0f)};
    for (uint block = 0u; block < INPUT_K; block += 256u) {
        value activation[VALUES];
        uint input = block + lane * VALUES;
        #pragma unroll
        for (short i = 0; i < VALUES; i++) {
            #pragma unroll
            for (short row = 0; row < ROWS; row++) {
                activation[i][row] = x[uint(row) * INPUT_K + input + uint(i)];
            }
        }
        #pragma unroll
        for (short column = 0; column < 4; column++) {
            value dot(0.0f);
            #pragma unroll
            for (short i = 0; i < VALUES; i++) {
                float weight = fp8_fetch(weights, first + uint(column), input + uint(i));
                dot += activation[i] * weight;
            }
            float scale = 1.0f;
            if constexpr (K4_SIDE == 1) {
                scale = fp8_scale(scales[(first + uint(column)) * (INPUT_K / 32) + block / 32u + lane / 4u]);
            }
            result[column] += scale * dot;
        }
    }
    #pragma unroll
    for (short column = 0; column < 4; column++) {
        value sum = simd_sum(result[column]);
        if (lane == 0u) {
            #pragma unroll
            for (short row = 0; row < ROWS; row++) {
                out[uint(row) * OUTPUT_N + first + uint(column)] = sum[row];
            }
        }
    }
}

// KERNEL_ENTRY
fp8_shared_rows<4>(weights, scales, xh, out, threadgroup_position_in_grid,
    simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
