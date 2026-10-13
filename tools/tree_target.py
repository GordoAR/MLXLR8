"""Private GLM tree verification; no proposal generation or production changes.

Node zero is the pending token at the saved prefix offset. Parents must precede
children. Node i attends the complete prefix, its ancestors, and itself; RoPE
uses prefix_offset + depth[i]. Current layer math and shared DSA selections stay
unchanged. Cache commit retains only root + accepted children; the bonus token
is still pending. Batched native arithmetic need not equal serial arithmetic.

Only batch-one, non-rotating KVCache groups from the main model are supported.
Pass mlx.core explicitly at GPU runtime, or NumPy for the CPU protocol tests.
"""

from dataclasses import dataclass
from operator import index

import numpy as np


@dataclass(frozen=True)
class TreeVerification:
    logits: object
    hidden: object  # Before the target final RMS norm.
    fused: object  # Requested layer inputs, in tap order, or None.
    topk: object  # Physical tree-cache IDs; valid before commit only.


class _LogicalRoPE:
    def __init__(self, original, positions, backend):
        self.original, self.positions, self.backend = original, positions, backend

    def __call__(self, x, offset=0):
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[2] != len(self.positions):
            raise ValueError("Tree RoPE requires all batch-one tree rows")
        # MLX RoPE accepts a vector of B offsets. Each node becomes a batch
        # example with T=1; heads and feature order are unchanged.
        offsets = self.backend.array(self.positions, dtype=self.backend.int32)
        return self.original(x.swapaxes(0, 2), offset=offsets).swapaxes(0, 2)


class TreeTarget:
    """One transactional tree forward; commit one connected path or roll back.

    Example: with TreeTarget(model, cache, [-1, 0, 0, 1], mx) as tree:
                 result = tree.verify(ids, tap=[2, 20, 39, 58, 75])
                 tree.commit([0, 1, 3])
    Do not use the same model concurrently while its RoPE objects are patched.
    """

    def __init__(self, model, cache, parents, backend):
        self.model, self.cache, self.backend = model, cache, backend
        self.parents = tuple(index(parent) for parent in parents)
        if not 1 <= len(self.parents) <= 32 or self.parents[0] != -1:
            raise ValueError("A tree needs one root and at most 32 nodes")
        depths = [0]
        ancestors = np.eye(len(self.parents), dtype=np.bool_)
        for node, parent in enumerate(self.parents[1:], 1):
            if not 0 <= parent < node:
                raise ValueError("A node parent must precede that node")
            depths.append(depths[parent] + 1)
            if depths[-1] > 8:
                raise ValueError("Tree depth must not exceed 8")
            ancestors[node] |= ancestors[parent]
        self.depths = tuple(depths)
        if len(cache) != len(model.layers) or not cache:
            raise ValueError("Tree verification needs one cache group per layer")
        self.members = []
        for layer, group in zip(model.layers, cache):
            expected = 2 if layer.self_attn.indexer is not None else 1
            if len(group.caches) != expected:
                raise ValueError("Cache group does not match the layer indexer")
            self.members.extend(group.caches)
        if any(type(member).__name__ != "KVCache" for member in self.members):
            raise ValueError("Only non-rotating KVCache is supported")
        self.offset = index(self.members[0].offset)
        if self.offset < 0 or any(member.offset != self.offset for member in self.members):
            raise ValueError("All main and indexer cache offsets must match")
        for member in self.members:
            if (member.keys is None) != (member.values is None):
                raise ValueError("Cache keys and values must exist together")
            if member.keys is None:
                if self.offset:
                    raise ValueError("Nonempty prefix needs cache tensors")
            elif any(value.ndim != 4 or value.shape[0] != 1
                     or value.shape[2] < self.offset for value in (member.keys, member.values)):
                raise ValueError("Tree cache tensors must be batch-one KV rows")
        self.positions = tuple(self.offset + depth for depth in self.depths)
        self.mask = np.concatenate((np.ones((len(self.parents), self.offset), dtype=np.bool_),
                                    ancestors), axis=-1)[None, None]
        self._entered = self._called = self._verified = self._committed = False
        self._patches = []

    def __enter__(self):
        if self._entered or self._called:
            raise ValueError("Use each TreeTarget context once")
        owners = [owner for layer in self.model.layers
                  for owner in (layer.self_attn, layer.self_attn.indexer) if owner is not None]
        if any(isinstance(owner.rope, _LogicalRoPE) for owner in owners):
            raise ValueError("Nested tree verification is not supported")
        self._snapshots = [(member.keys, member.values, member.offset) for member in self.members]
        for owner in owners:
            self._patches.append((owner, owner.rope))
            owner.rope = _LogicalRoPE(owner.rope, self.positions, self.backend)
        self._entered = True
        return self

    def verify(self, ids, tap=None):
        if not self._entered or self._called:
            raise ValueError("Verify once inside the TreeTarget context")
        if ids.shape != (1, len(self.parents)):
            raise ValueError("Inputs need one token for each tree node")
        tap = () if tap is None else tuple(index(layer) for layer in tap)
        if len(set(tap)) != len(tap) or any(not 0 <= layer < len(self.model.layers) for layer in tap):
            raise ValueError("Tap IDs must be unique layer-input positions")
        self._called = True
        h, topk, captured = self.model.embed_tokens(ids), None, {}
        mask = self.backend.array(self.mask)
        for layer_index, layer in enumerate(self.model.layers):
            if layer_index in tap:
                captured[layer_index] = h
            h, topk = layer(h, mask, self.cache[layer_index], topk)
            trace = getattr(self.model, "trace", None)
            if trace is not None:
                trace(layer_index, h)
        if any(member.offset != self.offset + len(self.parents) for member in self.members):
            raise ValueError("Tree forward did not update all cache rows")
        fused = self.backend.concatenate([captured[layer] for layer in tap], axis=-1) if tap else None
        result = TreeVerification(self.model.project_hidden(h), h, fused, topk)
        self._verified = True
        return result

    def commit(self, path):
        if not self._entered or not self._verified or self._committed:
            raise ValueError("Commit once after tree verification")
        path = tuple(index(node) for node in path)
        if not path or path[0] != 0 or any(not 0 <= node < len(self.parents) for node in path):
            raise ValueError("Accepted path must start at the root")
        if any(self.parents[child] != parent for parent, child in zip(path, path[1:])):
            raise ValueError("Accepted nodes must form one connected path")
        rows = self.backend.array([self.offset + node for node in path], dtype=self.backend.int32)
        end = self.offset + len(path)
        # Gather before assignment. Only appended rows are changed; the prefix
        # is neither gathered nor copied. Zero-width indexer values stay valid.
        for member in self.members:
            keys = self.backend.take(member.keys, rows, axis=2)
            values = self.backend.take(member.values, rows, axis=2)
            member.keys[..., self.offset:end, :] = keys
            member.values[..., self.offset:end, :] = values
            member.offset = end
        self._committed = True

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, original in reversed(self._patches):
            owner.rope = original
        if exc_type is not None or not self._committed:
            for member, state in zip(self.members, self._snapshots):
                member.keys, member.values, member.offset = state
        self._entered = False
