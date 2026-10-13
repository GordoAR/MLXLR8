// Exact float32 signed scaling and normalized block-local Hadamard.
uint block = thread_position_in_grid.x >> 5u;
uint lane = thread_position_in_grid.x & 31u;
uint row = threadgroup_position_in_grid.y;
uint projection = threadgroup_position_in_grid.z;
uint features = dims[0];
uint rows = dims[2];
if (block >= features / 128u) return;
uint expert = indices[row];
uint xrow = PER_SLOT ? row : row / dims[4];
float4 value;
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    ulong scale_index = ulong(expert) * features + col;
    float scale = projection ? float(in_up[scale_index]) : float(in_gate[scale_index]);
    value[j] = float(x[ulong(xrow) * features + col]) * scale;
}
value = exlr8_hadamard128(value, lane);
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    out[(ulong(projection) * rows + row) * features + col] = value[j];
}
