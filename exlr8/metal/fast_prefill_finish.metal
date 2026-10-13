// Normalized Hadamard128, output side scale, and inverse row gather.
// Transform: exllamav3 0.0.43 (MIT), Copyright (c) 2025 Turboderp.
uint block = thread_position_in_grid.x >> 5u;
uint lane = thread_position_in_grid.x & 31u;
uint row = threadgroup_position_in_grid.y;
uint features = dims[0];
if (block >= features / 128u) return;
uint expert = indices[row];
uint destination = order[row];
float4 value;
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    value[j] = inner[ulong(row) * features + col];
}
value = exlr8_hadamard128(value, lane);
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    out[ulong(destination) * features + col] = T(value[j] * float(out_scale[ulong(expert) * features + col]));
}
