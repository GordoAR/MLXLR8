uint linear = threadgroup_position_in_grid.z;
uint slot = (linear / BLOCK) % COUNT;
uint tile = (linear / (BLOCK * COUNT)) * BLOCK + linear % BLOCK;
if (tile >= OUTPUT_N / 8) return;
uint lane = thread_index_in_simdgroup;
uint out_row = tile * 8 + simdgroup_index_in_threadgroup * 4;
size_t expert = ids[slot];
const device uint8_t* ws = (const device uint8_t*)(weights + expert * OUTPUT_N * (INPUT_K / 8));
ws += out_row * (INPUT_K / 2) + lane * 8;
const device uint8_t* sl = scales + expert * OUTPUT_N * (INPUT_K / 32);
sl += out_row * (INPUT_K / 32) + lane / 2;
const device float* xp = x + size_t(lhs[slot]) * INPUT_K + lane * 16;
float values[16];
float result[4] = {0};
for (int k = 0; k < INPUT_K; k += 512) {
    #pragma unroll
    for (int i = 0; i < 16; i++) values[i] = xp[i];
    for (int row = 0; row < 4; row++) {
        uint8_t s = sl[row * (INPUT_K / 32)];
        uint32_t bits = s == 0 ? 0x400000 : (static_cast<uint16_t>(s) << 23);
        result[row] += expert_dot(ws + row * (INPUT_K / 2), values, as_type<float>(bits));
    }
    ws += 256;
    sl += 16;
    xp += 512;
}
for (int row = 0; row < 4; row++) {
    result[row] = simd_sum(result[row]);
    if (lane == 0) out[size_t(slot) * OUTPUT_N + out_row + row] = result[row];
}
