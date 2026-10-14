"""Load packed up1 experts into a bounded cache or resident buffers."""

from collections import OrderedDict
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.activations import swiglu

from .trellis import expert_linear
from .weights import SafeTensorFile


class ExpertStore:
    def __init__(self, path, cache_gib=32, resident=False, optimize=True):
        self.path = Path(path)
        self.resident = resident
        self.optimize = optimize
        self.manifest = json.loads(
            (self.path / "manifests/exlr8-manifest-up1.json").read_text()
        )
        if self.manifest["name"] != "up1":
            raise ValueError("This loader requires the up1 manifest")
        self.limit = int(cache_gib * 1024**3)
        if self.limit < 0:
            raise ValueError("cache_gib must be nonnegative")
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self._layers = {}

    def layer(self, number):
        if number not in self._layers:
            self._layers[number] = ExpertLayer(self, number)
        return self._layers[number]

    def _get(self, layer, expert):
        key = (layer.number, expert)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        shared = expert == 256
        prefix = "shared." if shared else ""
        start = 0 if shared else expert * layer.expert_bytes
        arrays = tuple(
            layer.weights.load_slice(
                prefix + "trellis." + projection, start, layer.expert_bytes
            )
            for projection in ("gate", "up", "down")
        )
        mx.eval(arrays)
        size = 3 * layer.expert_bytes
        if size <= self.limit:
            while self.cache_bytes + size > self.limit:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= sum(a.nbytes for a in old)
            self.cache[key] = arrays
            self.cache_bytes += size
        return arrays

    def stats(self):
        return {
            "cache_bytes": self.cache_bytes,
            "cache_limit_bytes": self.limit,
            "resident_bytes": self.resident_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }


class ExpertLayer:
    def __init__(self, store, number):
        self.store = store
        self.number = number
        plan = store.manifest["layers"][str(number)]
        widths = {v for rows in (plan["gate"], plan["up"]) for row in rows for v in row}
        widths.update(plan["down"])
        widths.update(plan["shared"]["gate"])
        widths.update(plan["shared"]["up"])
        widths.add(plan["shared"]["down"])
        self.k = 3 if number == 78 else 4
        if widths != {self.k}:
            raise ValueError(f"Layer {number} is not uniform K{self.k}")
        directory = store.path / "experts"
        self.weights = SafeTensorFile(directory / f"exlr8-layer-{number:03d}.k{self.k}.safetensors")
        index = SafeTensorFile(directory / f"exlr8-layer-{number:03d}.index.safetensors")
        self.hidden = index.header["signs.in"]["shape"][0]
        self.intermediate = index.header[f"k{self.k}.gate.svh"]["shape"][-1]
        self.expert_bytes = self.hidden * self.intermediate * self.k // 8
        self.affine = None
        self.scales = {}
        signs_in = index.raw("signs.in").astype(np.float16)
        signs_out = index.raw("signs.out").astype(np.float16)
        for projection in ("gate", "up", "down"):
            block_name = "sv_block" if projection == "down" else "su_block"
            channel_name = "suh" if projection == "down" else "svh"
            block = np.concatenate([
                index.raw(f"k{self.k}.{projection}.{block_name}"),
                index.raw(f"shared.k{self.k}.{projection}.{block_name}"),
            ])
            channel = np.concatenate([
                index.raw(f"k{self.k}.{projection}.{channel_name}"),
                index.raw(f"shared.k{self.k}.{projection}.{channel_name}"),
            ])
            hidden_scale = np.repeat(block, 128, axis=-1)
            hidden_scale *= signs_out if projection == "down" else signs_in
            self.scales[projection] = (
                (channel, hidden_scale)
                if projection == "down" else (hidden_scale, channel)
            )
        if store.resident:
            self.packed = {}
            for projection in ("gate", "up", "down"):
                packed = mx.concatenate([
                    self.weights.load("trellis." + projection),
                    self.weights.load("shared.trellis." + projection),
                ])
                mx.eval(packed)
                self.packed[projection] = packed
            self.scales = {
                projection: tuple(mx.array(scale) for scale in values)
                for projection, values in self.scales.items()
            }
            mx.eval(list(self.scales.values()))
            store.resident_bytes += sum(array.nbytes for array in self.packed.values())
            store.resident_bytes += sum(
                scale.nbytes for values in self.scales.values() for scale in values
            )
            if store.optimize and self.k == 4 and (self.hidden, self.intermediate) == (6144, 2048):
                from .native_affine import NativeAffine

                self.affine = NativeAffine(self)
                store.resident_bytes += (
                    sum(a.nbytes for values in self.affine.weights.values() for a in values)
                    - sum(a.nbytes for a in self.packed.values())
                )
                del self.packed
                mx.clear_cache()

    def __call__(self, x, ids):
        if self.affine is not None:
            return self.affine.slots(x, ids)
        if self.store.resident:
            remapped, packed, scales = ids, self.packed, self.scales
        else:
            # Routing must finish before reading the selected weight ranges.
            selected, inverse = np.unique(np.asarray(ids), return_inverse=True)
            if selected[0] < 0 or selected[-1] > 256:
                raise ValueError("Expert IDs must be in 0..256")
            remapped = mx.array(inverse.reshape(ids.shape), dtype=mx.uint32)
            weights = [self.store._get(self, int(expert)) for expert in selected]
            packed = {
                projection: mx.stack([w[p] for w in weights])
                for p, projection in enumerate(("gate", "up", "down"))
            }
            scales = {
                projection: tuple(mx.array(scale[selected]) for scale in values)
                for projection, values in self.scales.items()
            }
        if self.store.optimize:
            if ids.size // ids.shape[-1] <= 16:
                from .fast_decode import fast_expert_mlp as mlp
            else:
                from .fast_prefill import grouped_expert_mlp as mlp
            return mlp(x, remapped, packed, scales, self.k, self.hidden, self.intermediate)
        outputs = []
        for projection in ("gate", "up"):
            in_scale, out_scale = scales[projection]
            outputs.append(expert_linear(
                x, remapped, packed[projection], in_scale, out_scale,
                self.k, self.hidden, self.intermediate, sharded_side=1,
            ))
        hidden = swiglu(outputs[0], outputs[1])
        in_scale, out_scale = scales["down"]
        return expert_linear(
            hidden, remapped, packed["down"], in_scale, out_scale,
            self.k, self.intermediate, self.hidden, sharded_side=2,
        )
