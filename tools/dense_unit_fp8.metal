// Private dense GEMV. Packed uint32 bytes are in [output, input / 4] order.
// exlr8_fp8_byte converts each E4M3 code to exact half, then to float32.
// MXFP8 exponent 127 is required; there are no scale reads in this kernel.
uint tid = thread_position_in_threadgroup.x;
uint lane = tid & 31u;
uint simd = tid >> 5u;
uint first_output = (threadgroup_position_in_grid.x * 2u + simd) * OUTPUTS_PER_SIMD;
float accum[QUERY_ROWS][OUTPUTS_PER_SIMD] = {0.0f};

for (uint base = 0u; base < IN_FEATURES; base += 256u) {
    uint col = base + lane * 8u;
    float4 activation[QUERY_ROWS][2];
    #pragma unroll
    for (uint q = 0u; q < QUERY_ROWS; q++) {
        const device float4* xp = reinterpret_cast<const device float4*>(xh + q * IN_FEATURES + col);
        activation[q][0] = xp[0];
        activation[q][1] = xp[1];
    }
    #pragma unroll
    for (uint n = 0u; n < OUTPUTS_PER_SIMD; n++) {
        ulong index = ulong(first_output + n) * (IN_FEATURES / 4u) + col / 4u;
        uint2 words = uint2(packed[index], packed[index + 1u]);
        #pragma unroll
        for (uint p = 0u; p < 2u; p++) {
            uint word = words[p];
            float4 weight = float4(exlr8_fp8_byte(word & 255u),
                                  exlr8_fp8_byte((word >> 8u) & 255u),
                                  exlr8_fp8_byte((word >> 16u) & 255u),
                                  exlr8_fp8_byte(word >> 24u));
            #pragma unroll
            for (uint q = 0u; q < QUERY_ROWS; q++) {
                float4 a = activation[q][p];
                accum[q][n] = fma(a.x, weight.x, accum[q][n]);
                accum[q][n] = fma(a.y, weight.y, accum[q][n]);
                accum[q][n] = fma(a.z, weight.z, accum[q][n]);
                accum[q][n] = fma(a.w, weight.w, accum[q][n]);
            }
        }
    }
}

#pragma unroll
for (uint q = 0u; q < QUERY_ROWS; q++) {
    #pragma unroll
    for (uint n = 0u; n < OUTPUTS_PER_SIMD; n++) {
        float value = simd_sum(accum[q][n]);
        if (lane == 0u) out[q * OUT_FEATURES + first_output + n] = value;
    }
}
