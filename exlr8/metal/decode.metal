// EXL3 sliding 16-bit windows and tensor-core tile order (MIT),
// Copyright (c) 2025 Turboderp. See NOTICE and third_party/licenses/.
// Sources: exllamav3/exllamav3_ext/quant/exl3_dq.cuh and
// exllamav3/modules/quant/exl3_lib/quantize.py:tensor_core_perm.
// Modified for EXLR8 grouped128 storage and the FP8-rounded codebook.
#define PACKED_U32 (8u * K_BITS)
ulong index = thread_position_in_grid.x;
uint in_tiles = dims[0];
uint out_tiles = dims[1];
uint out_features = out_tiles * 16u;
if (index >= ulong(in_tiles * 16u) * out_features) {
    return;
}
uint r = uint(index / out_features);
uint c = uint(index % out_features);
uint tk = r / 16u;
uint tn = c / 16u;
uint lr = r & 15u;
uint lc = c & 15u;
uint lane = (lc & 7u) * 4u + ((lr & 7u) >> 1u);
uint slot = (lr & 1u) + ((lr & 8u) >> 2u) + ((lc & 8u) >> 1u);
uint code_index = lane * 8u + slot;
uint last_bit = (code_index + 1u) * K_BITS + 256u * K_BITS;
uint last_word = (last_bit - 1u) / 32u;
uint shift = (last_word + 1u) * 32u - last_bit;
ulong tile;
#if OUTPUT_GROUPED
tile = (ulong(tn / 8u) * in_tiles + tk) * 8u + tn % 8u;
#else
tile = ((ulong(tk / 8u) * (out_tiles / 8u) + tn / 8u) * 8u + tk % 8u) * 8u + tn % 8u;
#endif
const device uint* words = packed + tile * PACKED_U32;
ulong merged = (ulong(words[(last_word - 1u) % PACKED_U32]) << 32u) | ulong(words[last_word % PACKED_U32]);
uint cw = uint(merged >> shift) & 0xFFFFu;
out[index] = float(codebook[cw]);
