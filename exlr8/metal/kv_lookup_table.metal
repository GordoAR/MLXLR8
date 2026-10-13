uint i = thread_position_in_grid.x;
uint code = i & 255u;
uint magnitude = code & 127u;
if (magnitude == 127u) {
    table[i] = as_type<bfloat16_t>(ushort(0x7fc0));
} else {
    ushort bits = ushort((magnitude << 7) | (((magnitude + 1) >> 7) << 14)
                         | ((code >> 7) << 15));
    float value = float(as_type<half>(bits) * half(256));
    table[i] = bfloat16_t(value * scales[i >> 8]);
}
