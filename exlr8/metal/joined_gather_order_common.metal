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
// FP4 conversion and dot product derived from MLX 0.32.3.
#include <metal_stdlib>
using namespace metal;

inline float expert_fp4(uint8_t bits) {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return static_cast<float>(bits & 8 ? -converted : converted);
}

inline float expert_dot(const device uint8_t* w, const thread float* x, float scale) {
    float accum = 0;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < 4; i++) {
        accum += (x[4*i] * expert_fp4(ws[i]) +
                  x[4*i+1] * expert_fp4(ws[i] >> 4) +
                  x[4*i+2] * expert_fp4(ws[i] >> 8) +
                  x[4*i+3] * expert_fp4(ws[i] >> 12));
    }
    return scale * accum;
}
