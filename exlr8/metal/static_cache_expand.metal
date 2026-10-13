// EXL3 codeword windows (MIT), Copyright (c) 2025 Turboderp.
// See NOTICE and third_party/licenses/. Keep the original tile order.
#define PACKED_U32 (8u * K_BITS)
ulong index = thread_position_in_grid.x;
if (index >= lanes[0]) return;
ulong tile = index >> 5u;
uint lane = uint(index) & 31u;
const device uint* words = packed + tile * PACKED_U32;
for (uint g = 0u; g < 2u; g++) {
    uint value = 0u;
    for (uint q = 0u; q < 4u; q++) {
        uint j = g * 4u + q;
        uint last_bit = (lane * 8u + j + 1u) * K_BITS + 256u * K_BITS;
        uint last_word = (last_bit - 1u) / 32u;
        uint shift = (last_word + 1u) * 32u - last_bit;
        ulong merged = (ulong(words[(last_word - 1u) % PACKED_U32]) << 32u)
            | ulong(words[last_word % PACKED_U32]);
        uint cw = uint(merged >> shift) & 65535u;
        value |= uint(codebook[cw]) << (8u * q);
    }
    out[index * 2u + g] = value;
}
