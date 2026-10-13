"""CPU tree-mask/cache protocol checks; MLX smoke test needs a Metal lease."""

import copy
import os
from types import SimpleNamespace

import numpy as np
import pytest

from tools.tree_target import TreeTarget, _LogicalRoPE


class KVCache:
    def __init__(self, offset=3, width=2):
        self.offset = offset
        self.keys = np.arange(offset * width, dtype=np.float64).reshape(1, 1, offset, width)
        self.values = np.arange(offset, dtype=np.float64).reshape(1, 1, offset, 1)

    def update_and_fetch(self, keys, values):
        self.keys = np.concatenate((self.keys[:, :, :self.offset], keys), axis=2)
        self.values = np.concatenate((self.values[:, :, :self.offset], values), axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values


class RoPE:
    def __init__(self):
        self.calls = []

    def __call__(self, x, offset=0):
        offsets = np.asarray(offset)
        self.calls.append((x.shape, offsets.copy()))
        if offsets.ndim == 0:
            positions = offsets + np.arange(x.shape[2])[None]
        else:
            positions = offsets[:, None] + np.arange(x.shape[2])[None]
        return x + positions[:, None, :, None]


class Layer:
    def __init__(self, full=True):
        self.self_attn = SimpleNamespace(rope=RoPE(),
            indexer=SimpleNamespace(rope=RoPE()) if full else None)
        self.seen = []

    def __call__(self, h, mask, cache, previous):
        attention = self.self_attn
        keys = attention.rope(h[:, None])
        latent, _ = cache[0].update_and_fetch(keys, h[:, None, :, :1])
        if attention.indexer is not None:
            k = attention.indexer.rope(h[:, None])
            cache[1].update_and_fetch(k, h[:, None, :, :0])
            # Deliberately select every physical row, including siblings and
            # future descendants, so shared-layer masking is directly checked.
            topk = np.broadcast_to(np.arange(latent.shape[2]),
                                   (1, 1, h.shape[1], latent.shape[2])).copy()
        else:
            topk = previous
        valid = np.take_along_axis(mask, topk, axis=-1)
        physical = np.arange(latent.shape[2] - h.shape[1], latent.shape[2])
        valid &= topk <= physical[None, None, :, None]
        selected = np.take_along_axis(np.broadcast_to(latent[:, 0, None],
            (1, h.shape[1], latent.shape[2], h.shape[-1])), topk[:, 0, ..., None], axis=2)
        output = (selected * valid[:, 0, ..., None]).sum(axis=2) / valid[:, 0].sum(axis=-1)[..., None]
        self.seen.append((mask.copy(), topk.copy(), valid.copy()))
        return h + output * 0.01, topk


class Model:
    def __init__(self, full=(True, False)):
        self.layers = [Layer(value) for value in full]
        self.trace = None

    def embed_tokens(self, ids):
        return np.stack((ids, ids * 0.1), axis=-1).astype(np.float64)

    def project_hidden(self, h):
        return h @ np.array([[1., -1., 0.5], [0.2, 0.4, -0.3]])


class Group:
    def __init__(self, full, offset=3):
        self.caches = [KVCache(offset)]
        if full:
            indexer = KVCache(offset)
            indexer.values = np.zeros((1, 1, offset, 0))
            self.caches.append(indexer)

    def __getitem__(self, item):
        return self.caches[item]


def fixture(full=(True, False), offset=3):
    return Model(full), [Group(value, offset) for value in full]


def visible(cache):
    return [(member.offset, member.keys[..., :member.offset, :].copy(),
             member.values[..., :member.offset, :].copy())
            for group in cache for member in group.caches]


def assert_visible_equal(left, right):
    for (lo, lk, lv), (ro, rk, rv) in zip(left, right):
        assert lo == ro
        np.testing.assert_array_equal(lk, rk)
        np.testing.assert_array_equal(lv, rv)


def test_positions_and_ancestor_mask():
    model, cache = fixture()
    tree = TreeTarget(model, cache, [-1, 0, 0, 1, 2], np)
    assert tree.positions == (3, 4, 4, 5, 5)
    expected = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0],
                         [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=bool)
    np.testing.assert_array_equal(tree.mask[0, 0, :, :3], True)
    np.testing.assert_array_equal(tree.mask[0, 0, :, 3:], expected)


def test_vector_rope_matches_logical_one_row_calls():
    x = np.arange(2 * 5 * 4).reshape(1, 2, 5, 4)
    positions = (23860, 23861, 23861, 23862, 23862)
    actual = _LogicalRoPE(RoPE(), positions, np)(x, offset=23860)
    expected = np.concatenate([RoPE()(x[:, :, i:i+1], offset=position)
                               for i, position in enumerate(positions)], axis=2)
    np.testing.assert_array_equal(actual, expected)


def test_shared_topk_cannot_expose_siblings():
    model, cache = fixture()
    parents = [-1, 0, 0, 1, 2]
    with TreeTarget(model, cache, parents, np) as tree:
        tree.verify(np.array([[7, 8, 90, 10, 91]]))
        for layer in model.layers:
            np.testing.assert_array_equal(layer.seen[-1][2], tree.mask)
        assert model.layers[0].seen[-1][1] is not model.layers[1].seen[-1][1]
        np.testing.assert_array_equal(model.layers[0].seen[-1][1], model.layers[1].seen[-1][1])


def test_sibling_token_changes_do_not_change_other_path():
    model, cache = fixture()
    other, other_cache = copy.deepcopy((model, cache))
    parents = [-1, 0, 0, 1, 2]
    with TreeTarget(model, cache, parents, np) as tree:
        before = tree.verify(np.array([[7, 8, 90, 10, 91]]))
    with TreeTarget(other, other_cache, parents, np) as tree:
        after = tree.verify(np.array([[7, 8, 190, 10, 191]]))
    np.testing.assert_array_equal(before.hidden[:, [0, 1, 3]], after.hidden[:, [0, 1, 3]])


def test_branched_rows_match_isolated_path_and_cache():
    model, cache = fixture()
    serial, serial_cache = copy.deepcopy((model, cache))
    with TreeTarget(model, cache, [-1, 0, 0, 1, 2], np) as tree:
        result = tree.verify(np.array([[7, 8, 90, 10, 91]]), tap=[1, 0])
        branch_rows = [0, 1, 3]
        tree.commit(branch_rows)
    with TreeTarget(serial, serial_cache, [-1, 0, 1], np) as tree:
        reference = tree.verify(np.array([[7, 8, 10]]), tap=[1, 0])
        tree.commit([0, 1, 2])
    np.testing.assert_array_equal(result.hidden[:, branch_rows], reference.hidden)
    np.testing.assert_array_equal(result.logits[:, branch_rows], reference.logits)
    np.testing.assert_array_equal(result.fused[:, branch_rows], reference.fused)
    assert_visible_equal(visible(cache), visible(serial_cache))


def test_all_78_main_and_21_indexer_caches_commit_the_same_path():
    full = tuple(i < 3 or (i - 2) % 4 == 0 for i in range(78))
    model, cache = fixture(full)
    assert sum(full) == 21
    ropes = [owner.rope for layer in model.layers
             for owner in (layer.self_attn, layer.self_attn.indexer) if owner is not None]
    traces = []
    model.trace = lambda i, h: traces.append(i)
    with TreeTarget(model, cache, [-1, 0, 0, 1, 2], np) as tree:
        tree.verify(np.array([[7, 8, 90, 10, 91]]))
        expected = [(member.keys[:, :, [0, 1, 2, 3, 5, 7]].copy(),
                     member.values[:, :, [0, 1, 2, 3, 5, 7]].copy()) for member in tree.members]
        tree.commit([0, 2, 4])
        assert all(member.offset == 6 for member in tree.members)
        for member, (keys, values) in zip(tree.members, expected):
            np.testing.assert_array_equal(member.keys[:, :, :6], keys)
            np.testing.assert_array_equal(member.values[:, :, :6], values)
    restored = [owner.rope for layer in model.layers
                for owner in (layer.self_attn, layer.self_attn.indexer) if owner is not None]
    assert all(before is after for before, after in zip(ropes, restored))
    assert traces == list(range(78))


@pytest.mark.parametrize("failure", [False, True])
def test_uncommitted_or_failed_context_restores_rope_and_visible_cache(failure):
    model, cache = fixture()
    before, rope = visible(cache), model.layers[0].self_attn.rope
    try:
        with TreeTarget(model, cache, [-1, 0, 0], np) as tree:
            tree.verify(np.array([[7, 8, 90]]))
            if failure:
                tree.commit([0, 2])
                raise RuntimeError("after commit")
    except RuntimeError:
        pass
    assert_visible_equal(visible(cache), before)
    assert model.layers[0].self_attn.rope is rope


@pytest.mark.parametrize("parents", [[], [0], [-1, -1], [-1, 1], [-1, 2],
                                      [-1] + list(range(9)), [-1] + [0] * 32])
def test_invalid_or_unbounded_tree_is_rejected(parents):
    model, cache = fixture()
    with pytest.raises(ValueError):
        TreeTarget(model, cache, parents, np)


@pytest.mark.parametrize("path", [[], [1], [0, 2, 3], [0, 0], [0, 6]])
def test_invalid_commit_path_is_rejected(path):
    model, cache = fixture()
    with TreeTarget(model, cache, [-1, 0, 0, 1], np) as tree:
        tree.verify(np.array([[7, 8, 90, 10]]))
        with pytest.raises(ValueError):
            tree.commit(path)


def test_offsets_and_indexer_group_count_must_match():
    model, cache = fixture()
    cache[0][1].offset -= 1
    with pytest.raises(ValueError, match="offsets"):
        TreeTarget(model, cache, [-1], np)
    model, cache = fixture()
    cache[0].caches.pop()
    with pytest.raises(ValueError, match="indexer"):
        TreeTarget(model, cache, [-1], np)


def test_context_verify_and_tap_guards():
    model, cache = fixture()
    tree = TreeTarget(model, cache, [-1], np)
    with pytest.raises(ValueError):
        tree.verify(np.array([[7]]))
    with tree:
        with pytest.raises(ValueError):
            tree.verify(np.array([[7, 8]]))
        with pytest.raises(ValueError):
            tree.verify(np.array([[7]]), tap=[0, 0])
        tree.verify(np.array([[7]]), tap=[0])
        with pytest.raises(ValueError):
            tree.verify(np.array([[7]]))
        with pytest.raises(ValueError, match="Nested"):
            with TreeTarget(model, cache, [-1], np):
                pass


metal = pytest.mark.skipif(os.environ.get("EXLR8_TEST_METAL") != "1",
                          reason="set EXLR8_TEST_METAL=1 on a Metal host")


@metal
def test_mlx_logical_rope_vector_matches_scalar_rows():
    import mlx.core as mx
    import mlx.nn as nn

    rope = nn.RoPE(64, traditional=True, base=8000000)
    x = mx.array(np.random.default_rng(19).normal(size=(1, 3, 5, 128)), dtype=mx.bfloat16)
    positions = (23860, 23861, 23861, 23862, 23862)
    actual = _LogicalRoPE(rope, positions, mx)(x)
    expected = mx.concatenate([rope(x[:, :, i:i+1], offset=position)
                               for i, position in enumerate(positions)], axis=2)
    mx.eval(actual, expected)
    assert bool(mx.all(actual == expected))


@metal
def test_mlx_tiny_glm_tree_masks_positions_and_commit():
    import mlx.core as mx
    from exlr8.model import DecoderLayer, Model as GLMModel
    from test_model import fixtures

    args, weights, experts, vocab = fixtures(layers=2, topk=3)
    layers = [DecoderLayer(args, i, weights, None if i == 0 else experts) for i in range(2)]
    model = GLMModel(args, layers, vocab)
    prefix = model.make_cache()
    mx.eval(model(mx.array([list(range(2, 13))]), prefix))
    linear_ids = mx.array([list(range(1, 10))])
    native_cache, linear_cache = copy.deepcopy(prefix), copy.deepcopy(prefix)
    native_hidden, _ = model.forward_hidden(linear_ids, native_cache)
    native_logits = model.project_hidden(native_hidden)
    with TreeTarget(model, linear_cache, [-1] + list(range(8)), mx) as tree:
        linear = tree.verify(linear_ids)
        mx.eval(linear.logits, native_logits)
        assert bool(mx.all(linear.logits == native_logits))
        assert bool(mx.all(linear.hidden == native_hidden))
        tree.commit(list(range(9)))
    parents, path = [-1, 0, 0, 1, 2], [0, 1, 3]
    branched, isolated, changed = [copy.deepcopy(prefix) for _ in range(3)]
    with TreeTarget(model, branched, parents, mx) as tree:
        result = tree.verify(mx.array([[7, 8, 9, 10, 11]]), tap=[0, 1])
        mx.eval(result.logits, result.fused)
        tree.commit(path)
    with TreeTarget(model, isolated, [-1, 0, 1], mx) as tree:
        reference = tree.verify(mx.array([[7, 8, 10]]), tap=[0, 1])
        mx.eval(reference.logits, reference.fused)
        tree.commit([0, 1, 2])
    with TreeTarget(model, changed, parents, mx) as tree:
        perturbed = tree.verify(mx.array([[7, 8, 12, 10, 13]]), tap=[0, 1])
        mx.eval(perturbed.logits, perturbed.fused)
    rows = mx.array(path)
    # Equal geometry and changed siblings must leave the chosen path bit exact.
    assert bool(mx.all(mx.take(result.logits, rows, axis=1) == mx.take(perturbed.logits, rows, axis=1)))
    assert bool(mx.all(mx.take(result.fused, rows, axis=1) == mx.take(perturbed.fused, rows, axis=1)))
    # Different row counts can change native F32 reduction order. This check
    # proves only the tiny fixture's mask/position behavior, not full-model bits.
    np.testing.assert_allclose(np.asarray(mx.take(result.logits, rows, axis=1)),
                               np.asarray(reference.logits), rtol=2e-5, atol=2e-6)
    for tree_group, path_group in zip(branched, isolated):
        for tree_cache, path_cache in zip(tree_group.caches, path_group.caches):
            assert tree_cache.offset == path_cache.offset == 14
            np.testing.assert_allclose(np.asarray(tree_cache.keys[..., :14, :]),
                                       np.asarray(path_cache.keys[..., :14, :]), rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(np.asarray(tree_cache.values[..., :14, :]),
                                       np.asarray(path_cache.values[..., :14, :]), rtol=2e-5, atol=2e-6)
