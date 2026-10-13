// Sorted expert gather, signed side scale, and normalized Hadamard128.
// Transform: exllamav3 0.0.43 (MIT), Copyright (c) 2025 Turboderp.
uint block = thread_position_in_grid.x >> 5u;
uint lane = thread_position_in_grid.x & 31u;
uint row = threadgroup_position_in_grid.y;
uint projection = threadgroup_position_in_grid.z;
uint features = dims[0];
uint rows = dims[1];
if (block >= features / 128u) return;
float4 value = float4(0.0f);
if (row < rows) {
    uint expert = indices[row];
    uint source = order[row] / dims[2];
    for (uint j = 0u; j < 4u; j++) {
        uint col = block * 128u + lane + j * 32u;
        ulong si = ulong(expert) * features + col;
        float scale = projection == 0u ? float(in_gate[si]) : float(in_up[si]);
        value[j] = float(x[ulong(source) * features + col]) * scale;
    }
}
value = exlr8_hadamard128(value, lane);
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    out[(ulong(projection) * (rows + dims[3]) + row) * features + col] = value[j];
}
