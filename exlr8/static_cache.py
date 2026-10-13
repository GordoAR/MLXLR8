"""Fixed exact-E4M3 expert bytes in the original trellis tile order."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .trellis import _check_format, mcg_codebook


def _e4m3_codes(values):
    """Encode values already rounded to E4M3; do not change their values."""
    bits = np.asarray(values, np.float16).view(np.uint16).astype(np.uint32)
    exponent = (bits >> 10) & 31
    fraction = bits & 1023
    codes = (bits >> 8) & 128
    normal = exponent >= 9
    codes[normal] |= ((exponent[normal] - 8) << 3) | (fraction[normal] >> 7)
    subnormal = (exponent >= 6) & (exponent < 9)
    codes[subnormal] |= (1024 + fraction[subnormal]) >> (16 - exponent[subnormal])
    return codes.astype(np.uint8)


@lru_cache(maxsize=1)
def codebook_codes():
    return _e4m3_codes(mcg_codebook())


def expand_static_reference(packed, k, in_features, out_features, sharded_side=1):
    """Small CPU check: preserve raw tile order and byte 8*lane+j."""
    packed = np.asarray(packed)
    _check_format(packed, k, in_features, out_features, sharded_side)
    words = np.ascontiguousarray(packed).view(np.uint32).reshape(-1, 8 * k)
    end = (np.arange(256, dtype=np.uint32) + 1) * k + 256 * k
    last = (end - 1) // 32
    merged = (words[:, (last - 1) % (8 * k)].astype(np.uint64) << np.uint64(32))
    merged |= words[:, last % (8 * k)].astype(np.uint64)
    cw = (merged >> ((last + 1) * 32 - end).astype(np.uint64)) & 65535
    return codebook_codes()[cw].reshape(-1)


@lru_cache(maxsize=1)
def _device_codes():
    import mlx.core as mx

    return mx.array(codebook_codes())


@lru_cache(maxsize=3)
def _kernel(k):
    import mlx.core as mx

    source = (Path(__file__).with_name("metal") / "static_cache_expand.metal").read_text()
    return mx.fast.metal_kernel(
        name=f"exlr8_static_fp8_k{k}",
        input_names=["packed", "codebook", "lanes"], output_names=["out"],
        source=f"#define K_BITS {k}u\n" + source,
    )


def expand_static_trellis(packed, k, in_features, out_features, sharded_side=1):
    """Return flat exact FP8 bytes, with the stored side1/side2 tile order."""
    import mlx.core as mx

    experts = _check_format(packed, k, in_features, out_features, sharded_side)
    values = experts * in_features * out_features
    result = _kernel(k)(
        inputs=[packed.reshape(-1).view(mx.uint32), _device_codes(),
                mx.array([values // 8], mx.uint32)],
        grid=(values // 8, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(values // 4,)], output_dtypes=[mx.uint32],
    )[0]
    return result.view(mx.uint8)


def _expert_map(expert_ids, experts):
    ids = np.asarray(expert_ids)
    if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
        raise ValueError("hot expert IDs must be a nonempty integer vector")
    if np.any(ids < 0) or np.any(ids >= experts) or np.unique(ids).size != ids.size:
        raise ValueError("hot expert IDs must be unique and within the source layer")
    mapping = np.full(experts, 0xFFFFFFFF, np.uint32)
    mapping[ids] = np.arange(ids.size, dtype=np.uint32)
    return tuple(map(int, ids)), mapping


def static_cache_bytes(hidden, intermediate, hot_experts, experts=257):
    return 3 * hidden * intermediate * hot_experts + 4 * experts


class StaticExpertCache:
    """Expand fixed experts once. Decode uses the source scales and GPU map.

    ``fp8`` contains flat uint8 gate/up/down arrays. Each cache slot has
    ``hidden * intermediate`` bytes. Each original 16x16 tile has 256
    bytes; byte ``8*lane+j`` encodes the existing lane's codeword ``cw[j]``.
    ``expert_to_slot`` is uint32[E], with 0xFFFFFFFF for an uncached expert.
    """

    def __init__(self, layer, expert_ids=(256,)):
        import mlx.core as mx

        if not layer.store.resident:
            raise ValueError("static decode requires resident packed weights")
        self.layer = layer
        count = layer.scales["gate"][0].shape[0]
        self.expert_ids, mapping = _expert_map(expert_ids, count)
        self.expert_to_slot = mx.array(mapping)
        self.fp8 = {}
        size = layer.expert_bytes
        for projection in ("gate", "up", "down"):
            a, b, side = ((layer.intermediate, layer.hidden, 2) if projection == "down"
                          else (layer.hidden, layer.intermediate, 1))
            source = layer.packed[projection].reshape(-1)
            chunks = [expand_static_trellis(source[e * size:(e + 1) * size],
                                             layer.k, a, b, side)
                      for e in self.expert_ids]
            self.fp8[projection] = mx.concatenate(chunks)
            mx.eval(self.fp8[projection])
            del chunks
        mx.eval(self.expert_to_slot)

    def __call__(self, x, ids):
        from .fast_decode import fast_expert_mlp

        layer = self.layer
        return fast_expert_mlp(x, ids, layer.packed, layer.scales,
                               layer.k, layer.hidden, layer.intermediate,
                               fp8=self.fp8, expert_to_slot=self.expert_to_slot)

    def stats(self):
        return {"expert_ids": list(self.expert_ids),
                "weight_bytes": sum(a.nbytes for a in self.fp8.values()),
                "map_bytes": self.expert_to_slot.nbytes}
