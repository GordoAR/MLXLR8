// MCG16 codebook: exllamav3 0.0.43 (MIT), Copyright (c) 2025 Turboderp.
// EXLR8 adds round-to-nearest-even E4M3 after the mandatory half add.
// See NOTICE and third_party/licenses/.
inline float exlr8_decode(uint cw) {
    uint bits = ((cw * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half2 pair = as_type<half2>(bits);
    half sum = pair.x + pair.y;
    ushort h = as_type<ushort>(sum);
    ushort magnitude = h & ushort(0x7FFFu);
    ushort bias_bits = max(ushort((magnitude & ushort(0x7C00u)) + ushort(0x1C00u)), ushort(0x4000u));
    half bias = as_type<half>(bias_bits);
    half rounded_abs = as_type<half>(magnitude) + bias;
    return copysign(float(rounded_abs) - float(bias), float(sum));
}

inline float2 exlr8_decode2(uint2 cw) {
    uint2 bits = ((cw * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half4 pair = as_type<half4>(bits);
    half2 sum = pair.xz + pair.yw;
    ushort2 h = as_type<ushort2>(sum);
    ushort2 magnitude = h & ushort(0x7FFFu);
    ushort2 bias_bits = max(ushort2((magnitude & ushort(0x7C00u)) + ushort(0x1C00u)), ushort2(0x4000u));
    half2 bias = as_type<half2>(bias_bits);
    half2 rounded_abs = as_type<half2>(magnitude) + bias;
    return copysign(float2(rounded_abs) - float2(bias), float2(sum));
}

inline float4 exlr8_hadamard128(float4 value, uint lane) {
    for (uint bit = 1u; bit < 32u; bit <<= 1u) {
        float4 other = simd_shuffle_xor(value, ushort(bit));
        value = (lane & bit) ? other - value : value + other;
    }
    value = float4(value.x + value.y, value.x - value.y,
                   value.z + value.w, value.z - value.w);
    value = float4(value.x + value.z, value.y + value.w,
                   value.x - value.z, value.y - value.w);
    return value * 0.08838834764831845f;
}

template <typename T>
inline float exlr8_swiglu(float gate, float up) {
    // MLX Sigmoid<T>: Copyright 2023 Apple Inc. (MIT).
    // Keep typed exp, division, and multiplication at the existing handoff.
    T g = T(gate);
    T u = T(up);
    auto tail = 1 / (1 + precise::exp(abs(g)));
    T sigmoid = g < 0 ? tail : 1 - tail;
    return float(T(T(g * sigmoid) * u));
}
