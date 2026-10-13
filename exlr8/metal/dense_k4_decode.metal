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
// MLX fp_quantized.h: Copyright © 2025-2026 Apple Inc.
// Decoder helpers and matrix-vector template adapted from MLX 0.32.3.
// Source: https://github.com/ml-explore/mlx/tree/v0.32.3/mlx/backend/metal/kernels
// Changes: remove unused declarations and encoder constructors; replace FP8
// weight loads with the tested EXLR8 K4 fetch. Arithmetic order is unchanged.
// fp_quantized.h SHA256 359a1ea9008f74cc15c61e9d39c6300dcffb9cfb9cc0ebd59ec7c00b789d719f
// fp8.h SHA256 9a9d77a6295f66b4ada4311c3b855b6ecb7a2b38f28623109e3c61d2e17e81ae
// fp4.h SHA256 e3159945d7e5ae24f7a1ca9c4875b2a5d15cfb6952405df24a0df1bb2622a93a
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// BEGIN fp4_constants
constant constexpr float F8E4M3_MAX = 448.0f;
constant constexpr float F4E2M1_MAX = 6.0f;
// END fp4_constants

// BEGIN fp4_decoder
struct fp4_e2m1 {
  operator float16_t() thread {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return bits & 8 ? -converted : converted;
  }

  operator float() thread {
    return static_cast<float>(this->operator float16_t());
  }

  operator bfloat16_t() thread {
    return static_cast<bfloat16_t>(this->operator float16_t());
  }

  uint8_t bits;
};
// END fp4_decoder

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
// END fp8_e4m3_decoder

// BEGIN fp8_e8m0_decoder
struct fp8_e8m0 {
  operator bfloat16_t() thread {
    uint16_t out = (bits == 0 ? 0x40 : (static_cast<uint16_t>(bits) << 7));
    return as_type<bfloat16_t>(out);
  }

  operator float() thread {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }

  uint8_t bits;
};
// END fp8_e8m0_decoder

// BEGIN simd_constant
#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
// END simd_constant

// BEGIN packing_and_scale
template <int wsize = 8, int bits = 4>
inline constexpr short get_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    // Use nv scale
    return T(*(thread fp8_e4m3*)(&s));
  } else {
    return T(*(thread fp8_e8m0*)(&s));
  }
}
// END packing_and_scale

// BEGIN dequantize_and_load
template <int bits, typename U = float>
struct Dequantize {
  U operator()(uint8_t x) thread {
    if constexpr (bits == 8) {
      return U(*(thread fp8_e4m3*)(&x));
    } else {
      return U(*(thread fp4_e2m1*)(&x));
    }
  }
};

template <typename T, typename U, int values_per_thread>
inline void load_vector(const device T* x, thread U* x_thread) {
#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    x_thread[i] = x[i];
  }
}
// END dequantize_and_load

// BEGIN fetch
// Private exact K4 fetch. Tile and bit addresses follow dense_fp8_decode.metal.
inline float k4_fetch(const device uint* packed, uint c, uint r) {
    uint tn = c / 16u, tk = r / 16u;
    uint lc = c & 15u, lr = r & 15u;
    uint lane = (lc & 7u) * 4u + ((lr & 7u) >> 1u);
    uint slot = (lr & 1u) + ((lr & 8u) >> 2u) + ((lc & 8u) >> 1u);
    uint last_bit = (lane * 8u + slot + 1u) * 4u + 1024u;
    uint last_word = (last_bit - 1u) / 32u;
    uint shift = (last_word + 1u) * 32u - last_bit;
    ulong tile;
    if constexpr (K4_SIDE == 1) {
        tile = (ulong(tn / 8u) * (INPUT_K / 16u) + tk) * 8u + tn % 8u;
    } else {
        tile = ((ulong(tk / 8u) * (OUTPUT_N / 128u) + tn / 8u) * 8u + tk % 8u) * 8u + tn % 8u;
    }
    const device uint* words = packed + tile * 32u;
    ulong merged = (ulong(words[(last_word - 1u) % 32u]) << 32u) | ulong(words[last_word % 32u]);
    return exlr8_decode(uint(merged >> shift) & 65535u);
}

// END fetch

// BEGIN dot
template <typename U, int values_per_thread, int bits>
inline U k4_qdot(const device uint32_t* w, const thread U* x_thread, U scale, uint output_row, uint input_start) {
  U accum = 0;
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * Dequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * Dequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * Dequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * Dequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * k4_fetch(w, output_row, input_start + uint(i));
    }
  }

  return scale * accum;
}

// END dot

// BEGIN core
template <
    typename T,
    int group_size,
    int bits,
    bool has_global_scale = false,
    int results_per_simdgroup = 4>
METAL_FUNC void k4_qmv_fast_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  // K4 keeps the packed base.

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  // K4 uses absolute output and input indices.
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = w;
      const device auto* sl = scales + row * in_vec_size_g;

      U s = dequantize_scale<U, group_size>(sl[0]);
      result[row] += k4_qdot<U, values_per_thread, bits>(wl, x_thread, s, out_row + row, k + simd_lid * values_per_thread);
    }

    // K4 advances through k in its fetch.
    scales += block_size / group_size;
    x += block_size;
  }

  float inv_scale_enc = 1.0f;
  if constexpr (has_global_scale) {
    inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      if constexpr (has_global_scale) {
        y[row] = static_cast<T>(result[row] * inv_scale_enc);
      } else {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

// END core

// Batched projection reuses each decoded K4 value across token rows.
// Decoder helpers come from the licensed production dense_k4_decode header.
template <short ROWS>
METAL_FUNC void k4_shared_rows(
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
                float weight = k4_fetch(weights, first + uint(column), input + uint(i));
                dot += activation[i] * weight;
            }
            float scale = 1.0f;
            if constexpr (K4_SIDE == 1) {
                scale = dequantize_scale<float, 32>(scales[(first + uint(column)) * (INPUT_K / 32) + block / 32u + lane / 4u]);
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
if constexpr (BATCH_ROWS == 1) {
  k4_qmv_fast_impl<float, 32, 8, false, 4>(
      weights, scales, nullptr, xh, out, INPUT_K, OUTPUT_N,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
} else {
  k4_shared_rows<BATCH_ROWS>(weights, scales, xh, out,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
}
