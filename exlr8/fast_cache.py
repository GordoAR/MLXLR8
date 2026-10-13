"""Optional exact-FP8 expert cache for sequential decode experiments.

Each slot stores gate, up, and down E4M3 codes. The E8M0 exponent is
always 127, so two exponent matrices are broadcast across all slots.
On a miss the prior output is evaluated before a slot is replaced. This
lets MLX reuse private, contiguous buffers. Retaining unrelated lazy
outputs can prevent reuse and force MLX to copy a layer buffer.
"""

from collections import OrderedDict, deque
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


def cache_bytes(hidden, intermediate, slots):
    """Allocated weight bytes, including F32 side vectors and exponents."""
    codes = 3 * hidden * intermediate * slots
    side_vectors = 3 * (hidden + intermediate) * 4 * slots
    exponents = 2 * hidden * intermediate // 32
    return codes + side_vectors + exponents


class RouteTrace:
    """Retain decode IDs; compute hit rates after the timed run finishes.

    Call ``record(layer.number, ids)`` from an external ExpertLayer hook.
    Recording uses shape metadata only. It does not read GPU route values.
    Prefill routes with more than one token are omitted.
    """

    def __init__(self, token_limit=512, capacities=(12, 18)):
        if token_limit <= 0:
            raise ValueError("token_limit must be positive")
        self.token_limit = token_limit
        self.capacities = capacities
        self.records = {}

    def record(self, layer_number, ids):
        if np.prod(ids.shape[:-1]) != 1:
            return
        rows = self.records.setdefault(layer_number, deque(maxlen=self.token_limit))
        rows.append(ids)

    def to_dict(self):
        result = {"token_limit": self.token_limit, "layers": {}, "totals": {}}
        for number, records in sorted(self.records.items()):
            routes = [np.asarray(ids).reshape(-1) for ids in records]
            layer = {"routes": [row.tolist() for row in routes], "caches": {}}
            for capacity in self.capacities:
                cache = _SlotLRU(capacity)
                warm_tokens = 0
                for row in routes:
                    _, missing = cache.select(row)
                    warm_tokens += not missing
                row = {"hits": cache.hits, "misses": cache.misses,
                       "tokens": len(routes), "all_selected_hit_tokens": warm_tokens}
                layer["caches"][str(capacity)] = row
                total = result["totals"].setdefault(str(capacity), dict.fromkeys(row, 0))
                for key, value in row.items():
                    total[key] += value
            result["layers"][str(number)] = layer
        for row in result["totals"].values():
            row["hit_rate"] = row["hits"] / (row["hits"] + row["misses"])
            row["all_selected_hit_token_rate"] = row["all_selected_hit_tokens"] / row["tokens"]
        return result

    def write(self, path):
        result = self.to_dict()
        Path(path).write_text(json.dumps(result, indent=2) + "\n")
        return result


class _SlotLRU:
    def __init__(self, capacity, experts=257):
        if not isinstance(capacity, int) or not 1 <= capacity <= experts:
            raise ValueError("cache slots must be in 1..expert_count")
        self.capacity = capacity
        self.experts = experts
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def select(self, ids):
        ids = np.asarray(ids)
        if ids.size == 0 or ids.dtype.kind not in "iu":
            raise ValueError("expert IDs must be a nonempty integer array")
        selected = np.unique(ids)
        if selected[0] < 0 or selected[-1] >= self.experts:
            raise ValueError("expert ID is outside the source layer")
        if selected.size > self.capacity:
            raise ValueError("selected experts exceed the cache slot count")
        needed = set(map(int, selected))
        free = set(range(self.capacity)).difference(self.entries.values())
        missing = []
        for value in selected:
            expert = int(value)
            if expert in self.entries:
                self.hits += 1
            else:
                self.misses += 1
                if free:
                    slot = min(free)
                    free.remove(slot)
                else:
                    victim = next(e for e in self.entries if e not in needed)
                    slot = self.entries.pop(victim)
                    self.evictions += 1
                self.entries[expert] = slot
                missing.append((expert, slot))
            self.entries.move_to_end(expert)
        remapped = np.array([self.entries[int(e)] for e in ids.flat], np.uint32)
        return remapped.reshape(ids.shape), missing


def native_expert_mlp(x, indices, gate_up, down, gate_exponents, down_exponents,
                      in_gate, out_gate, in_up, out_up, in_down, out_down):
    """Native FP8 gather with the current F32 transforms and typed handoffs."""
    import mlx.core as mx

    from .fast_decode import _dims, _finish, _input_rows, _prep, _transform_kernel

    slots, _, intermediate, hidden_packs = gate_up.shape
    hidden = hidden_packs * 4
    per_slot = _input_rows(x, indices, hidden)
    ids = indices.reshape(-1).astype(mx.uint32)
    rows = ids.size
    xh = _prep(x, ids, in_gate, in_up, hidden, indices.shape[-1], per_slot, 2)
    gate_ids = mx.concatenate([ids * 2, ids * 2 + 1])
    inner = mx.gather_qmm(
        xh.reshape(2 * rows, 1, hidden),
        gate_up.reshape(2 * slots, intermediate, hidden_packs),
        scales=gate_exponents, rhs_indices=gate_ids, mode="mxfp8",
    ).reshape(2 * rows, 1, intermediate)
    down_xh = _transform_kernel("gate_finish")(
        inputs=[inner, ids, out_gate, out_up, in_down,
                _dims(intermediate, 0, rows, 1)],
        template=[("T", x.dtype)],
        grid=(intermediate // 128 * 32, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, intermediate)],
        output_dtypes=[mx.float32],
    )[0]
    inner = mx.gather_qmm(
        down_xh.reshape(rows, 1, intermediate), down,
        scales=down_exponents, rhs_indices=ids, mode="mxfp8",
    ).reshape(rows, 1, hidden)
    return _finish(inner, ids, out_down, hidden, 1, x.dtype).reshape(
        indices.shape + (hidden,)
    )


@lru_cache(maxsize=1)
def _compiled_mlp():
    import mlx.core as mx

    return mx.compile(native_expert_mlp)


class FastExpertCache:
    """Private per-layer LRU; the existing ExpertLayer path is unchanged.

    ``layer`` is an ExpertLayer. Its packed source may be resident or file
    backed. A file miss reads three expert ranges directly, without adding
    them to ExpertStore's separate packed cache. Routing converts IDs to
    NumPy and thus synchronizes a GPU-produced route with the host.
    """

    def __init__(self, layer, slots=12):
        import mlx.core as mx

        self.layer = layer
        hidden, intermediate = layer.hidden, layer.intermediate
        self._lru = _SlotLRU(slots, layer.scales["gate"][0].shape[0])
        self._gate_up = mx.zeros((slots, 2, intermediate, hidden // 4), mx.uint32)
        self._down = mx.zeros((slots, hidden, intermediate // 4), mx.uint32)
        self._gate_exponents = mx.broadcast_to(
            mx.full((intermediate, hidden // 32), 127, mx.uint8),
            (2 * slots, intermediate, hidden // 32),
        )
        self._down_exponents = mx.broadcast_to(
            mx.full((hidden, intermediate // 32), 127, mx.uint8),
            (slots, hidden, intermediate // 32),
        )
        self._scales = {
            p: (mx.zeros((slots, a), mx.float32), mx.zeros((slots, b), mx.float32))
            for p, a, b in (("gate", hidden, intermediate),
                            ("up", hidden, intermediate),
                            ("down", intermediate, hidden))
        }
        self._pending = None
        self._call = _compiled_mlp()
        mx.eval(self._gate_up, self._down, self._gate_exponents,
                self._down_exponents, self._scales)

    def _packed(self, expert, projection):
        size = self.layer.expert_bytes
        if self.layer.store.resident:
            return self.layer.packed[projection].reshape(-1)[expert * size:(expert + 1) * size]
        shared = expert == 256
        return self.layer.weights.load_slice(
            ("shared." if shared else "") + "trellis." + projection,
            0 if shared else expert * size, size,
        )

    def prepare(self, ids):
        """Resolve host IDs and fill misses; return current slot indices."""
        import mlx.core as mx

        from .dense_trellis import expand_trellis

        remapped, missing = self._lru.select(ids)
        if missing and self._pending is not None:
            mx.eval(self._pending)
            self._pending = None
        h, m, k = self.layer.hidden, self.layer.intermediate, self.layer.k
        for expert, slot in missing:
            gate, _ = expand_trellis(self._packed(expert, "gate"), k, h, m, 1)
            up, _ = expand_trellis(self._packed(expert, "up"), k, h, m, 1)
            down, _ = expand_trellis(self._packed(expert, "down"), k, m, h, 2)
            self._gate_up[slot] = mx.stack([gate, up])
            self._down[slot] = down
            for projection, vectors in self._scales.items():
                for source, target in zip(self.layer.scales[projection], vectors):
                    target[slot] = mx.array(source[expert], dtype=mx.float32)
            mx.eval(self._gate_up, self._down, self._scales)
        return mx.array(remapped)

    def _apply(self, x, slots):
        """Queue a call; slots must come from the current cache mapping."""
        vectors = [v for p in ("gate", "up", "down") for v in self._scales[p]]
        self._pending = self._call(
            x, slots, self._gate_up, self._down,
            self._gate_exponents, self._down_exponents, *vectors,
        )
        return self._pending

    def __call__(self, x, ids):
        return self._apply(x, self.prepare(ids))

    def stats(self):
        return {
            "slots": self._lru.capacity,
            "cached_experts": len(self._lru.entries),
            "weight_bytes": cache_bytes(self.layer.hidden, self.layer.intermediate,
                                         self._lru.capacity),
            "hits": self._lru.hits,
            "misses": self._lru.misses,
            "evictions": self._lru.evictions,
        }
