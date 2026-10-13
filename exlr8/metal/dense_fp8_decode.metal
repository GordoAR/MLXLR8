// EXL3 window and tile geometry (MIT), Copyright (c) 2025 Turboderp.
// See NOTICE and third_party/licenses/. Output codes keep E4M3 values exactly.
#define PACKED_U32 (8u * K_BITS)
ulong index = thread_position_in_grid.x;
uint in_tiles = dims[0];
uint out_tiles = dims[1];
uint in_features = in_tiles * 16u;
uint out_features = out_tiles * 16u;
if (index >= ulong(in_features / 4u) * out_features) return;
uint c = uint(index / (in_features / 4u));
uint r_start = uint(index % (in_features / 4u)) * 4u;
uint result = 0u;
for (uint j = 0u; j < 4u; j++) {
    uint r = r_start + j;
    uint tk = r / 16u;
    uint tn = c / 16u;
    uint lr = r & 15u;
    uint lc = c & 15u;
    uint lane = (lc & 7u) * 4u + ((lr & 7u) >> 1u);
    uint slot = (lr & 1u) + ((lr & 8u) >> 2u) + ((lc & 8u) >> 1u);
    uint last_bit = (lane * 8u + slot + 1u) * K_BITS + 256u * K_BITS;
    uint last_word = (last_bit - 1u) / 32u;
    uint shift = (last_word + 1u) * 32u - last_bit;
    ulong tile;
#if OUTPUT_GROUPED
    tile = (ulong(tn / 8u) * in_tiles + tk) * 8u + tn % 8u;
#else
    tile = ((ulong(tk / 8u) * (out_tiles / 8u) + tn / 8u) * 8u + tk % 8u) * 8u + tn % 8u;
#endif
    const device uint* words = packed + tile * PACKED_U32;
    ulong merged = (ulong(words[(last_word - 1u) % PACKED_U32]) << 32u)
        | ulong(words[last_word % PACKED_U32]);
    uint cw = uint(merged >> shift) & 65535u;
    ushort bits = as_type<ushort>(codebook[cw]);
    uint exponent = (uint(bits) >> 10u) & 31u;
    uint fraction = uint(bits) & 1023u;
    uint code = (uint(bits) >> 8u) & 128u;
    if (exponent >= 9u) {
        code |= ((exponent - 8u) << 3u) | (fraction >> 7u);
    } else if (exponent >= 6u) {
        code |= (1024u + fraction) >> (16u - exponent);
    }
    result |= code << (8u * j);
}
out[index] = result;
