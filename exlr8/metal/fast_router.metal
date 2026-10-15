uint lane = thread_index_in_simdgroup;
uint token = threadgroup_position_in_grid.x;
float original[8], choice[8];
#pragma unroll
for (uint i = 0; i < 8; ++i) {
    uint expert = i * 32 + lane;
    float value = logits[token * 256 + expert];
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(value)));
    original[i] = value < 0 ? y : 1.0f - y;
    choice[i] = original[i] + bias[expert];
}
float selected[8];
#pragma unroll
for (uint rank = 0; rank < 8; ++rank) {
    float best = -INFINITY;
    uint index = 256;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        uint expert = i * 32 + lane;
        if (choice[i] > best || (choice[i] == best && expert < index)) {
            best = choice[i];
            index = expert;
        }
    }
    float winning_value = simd_max(best);
    uint winner = simd_min(best == winning_value ? index : 256u);
    uint local_slot = winner / 32;
    float weight = simd_broadcast(original[local_slot], winner % 32);
    if (lane == 0) {
        indices[token * 9 + rank] = winner;
        selected[rank] = weight;
    }
    if (lane == winner % 32) choice[local_slot] = -INFINITY;
}
if (lane == 0) {
    float total = 1e-20f;
    #pragma unroll
    for (uint i = 0; i < 8; ++i) total += selected[i];
    #pragma unroll
    for (uint i = 0; i < 8; ++i) scores[token * 8 + i] = selected[i] / total * 2.5f;
    indices[token * 9 + 8] = 256;
}
