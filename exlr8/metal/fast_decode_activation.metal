// Exact row activations from aligned float2 loads. Each accumulator keeps
// the same tile order and float32 FMA as the shuffle path.
inline void exlr8_direct_fma(thread float* accum,
                             const thread float* weight,
                             const device float* xh,
                             uint row, uint features, uint tile, uint lane) {
    ulong base = ulong(row) * features + tile * 16u + (lane & 3u) * 2u;
    #pragma unroll
    for (uint j = 0u; j < 4u; j += 2u) {
        float2 activation = *reinterpret_cast<const device float2*>(xh + base + j * 4u);
        accum[j] = fma(activation.x, weight[j], accum[j]);
        accum[j + 4u] = fma(activation.x, weight[j + 4u], accum[j + 4u]);
        accum[j + 1u] = fma(activation.y, weight[j + 1u], accum[j + 1u]);
        accum[j + 5u] = fma(activation.y, weight[j + 5u], accum[j + 5u]);
    }
}

template <typename V, typename I, uint Width>
inline void exlr8_direct_fma(thread V* accum,
                             const thread float* weight,
                             const device float* xh,
                             I rows, uint count, uint features, uint tile, uint lane) {
    #pragma unroll
    for (uint r = 0u; r < Width; r++) {
        if (r < count) {
            ulong base = ulong(rows[r]) * features + tile * 16u + (lane & 3u) * 2u;
            #pragma unroll
            for (uint j = 0u; j < 4u; j += 2u) {
                float2 activation = *reinterpret_cast<const device float2*>(xh + base + j * 4u);
                accum[j][r] = fma(activation.x, weight[j], accum[j][r]);
                accum[j + 4u][r] = fma(activation.x, weight[j + 4u], accum[j + 4u][r]);
                accum[j + 1u][r] = fma(activation.y, weight[j + 1u], accum[j + 1u][r]);
                accum[j + 5u][r] = fma(activation.y, weight[j + 5u], accum[j + 5u][r]);
            }
        }
    }
}
