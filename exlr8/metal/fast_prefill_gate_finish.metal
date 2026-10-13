// Gate/up post-transform, typed SwiGLU, and down pre-transform.
// Transform: exllamav3 0.0.43 (MIT), Copyright (c) 2025 Turboderp.
// Typed activation: MLX (MIT), Copyright 2023-2024 Apple Inc.
uint block = thread_position_in_grid.x >> 5u;
uint lane = thread_position_in_grid.x & 31u;
uint row = threadgroup_position_in_grid.y;
uint features = dims[0];
uint rows = dims[1];
if (block >= features / 128u) return;
float4 hidden = float4(0.0f);
if (row < rows) {
    uint expert = indices[row];
    float4 gate, up;
    for (uint j = 0u; j < 4u; j++) {
        uint col = block * 128u + lane + j * 32u;
        gate[j] = inner[ulong(row) * features + col];
        up[j] = inner[(ulong(rows) + row) * features + col];
    }
    gate = exlr8_hadamard128(gate, lane);
    up = exlr8_hadamard128(up, lane);
    for (uint j = 0u; j < 4u; j++) {
        uint col = block * 128u + lane + j * 32u;
        ulong si = ulong(expert) * features + col;
        T g = T(gate[j] * float(out_gate[si]));
        T u = T(up[j] * float(out_up[si]));
        // Match MLX's typed Sigmoid and the two typed multiplications.
        auto tail = 1 / (1 + precise::exp(abs(g)));
        T sigmoid = g < T(0) ? tail : 1 - tail;
        hidden[j] = float(T(T(g * sigmoid) * u));
        hidden[j] *= float(in_down[si]);
    }
    hidden = exlr8_hadamard128(hidden, lane);
}
for (uint j = 0u; j < 4u; j++) {
    uint col = block * 128u + lane + j * 32u;
    out[ulong(row) * features + col] = hidden[j];
}
