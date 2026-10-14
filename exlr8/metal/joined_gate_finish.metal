// Fused gate/up split sum, post-Hadamard, SwiGLU, down signed scale,
// and down pre-Hadamard. All transforms and inner sums stay float32.
uint block = thread_position_in_grid.x >> 5u;
uint lane = thread_position_in_grid.x & 31u;
uint row = threadgroup_position_in_grid.y;
uint features = dims[0];
if (block >= features / 128u) return;
uint expert = indices[row];
float4 gate = float4(0.0f);
float4 up = float4(0.0f);
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    gate[j] = inner[(ulong(row) * 2u) * features + col];
    up[j] = inner[(ulong(row) * 2u + 1u) * features + col];
}
gate = exlr8_hadamard128(gate, lane);
up = exlr8_hadamard128(up, lane);
float4 hidden;
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    ulong scale_index = ulong(expert) * features + col;
    hidden[j] = exlr8_swiglu<T>(gate[j] * float(out_gate[scale_index]),
                              up[j] * float(out_up[scale_index]));
    hidden[j] *= float(in_down[scale_index]);
}
hidden = exlr8_hadamard128(hidden, lane);
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    out[ulong(row) * features + col] = hidden[j];
}
