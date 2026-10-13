// Exact float32 split sum, normalized Hadamard, and signed output scale.
uint block = thread_position_in_grid.x >> 5u;
uint lane = thread_position_in_grid.x & 31u;
uint row = threadgroup_position_in_grid.y;
uint features = dims[0];
uint splits = dims[3];
if (block >= features / 128u) return;
uint expert = indices[row];
float4 value = float4(0.0f);
for (uint s = 0u; s < splits; s++) {
    for (uint j = 0u; j < 4u; j++) {
        uint col = block * 128u + lane + j * 32u;
        value[j] += inner[(ulong(row) * splits + s) * features + col];
    }
}
value = exlr8_hadamard128(value, lane);
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    out[ulong(row) * features + col] = T(value[j] * float(out_scale[ulong(expert) * features + col]));
}
