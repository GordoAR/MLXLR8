"""CPU proofs for private proposal selection and greedy tree acceptance."""

import numpy as np
import pytest

from types import SimpleNamespace

from tools.dspark_tree_proposals import ProposalTree, accepted_path, build_tree, draft_tree


def rows(previous, position, width):
    ids = np.array([[token * 10 + rank + 1 for rank in range(width)] for token in previous])
    logp = np.broadcast_to(-np.arange(width, dtype=float) - position / 10, ids.shape)
    return ids, logp


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("depth", [1, 4, 8])
def test_bounded_topological_tree_keeps_greedy_path(width, depth):
    tree = build_tree(0, rows, depth, width)
    assert len(tree.tokens) == 1 + width * depth <= 25
    assert tree.parents[0] == -1
    assert all(0 <= parent < node for node, parent in enumerate(tree.parents[1:], 1))
    assert all(tree.depths[node] == tree.depths[parent] + 1
               for node, parent in enumerate(tree.parents[1:], 1))
    assert len(tree.greedy_path) == depth + 1
    token = 0
    for node in tree.greedy_path[1:]:
        token = token * 10 + 1
        assert tree.tokens[node] == token


def test_greedy_trunk_survives_lower_beam_score():
    def lower(previous, position, width):
        ids, scores = rows(previous, position, width)
        if position:
            scores = scores.copy()
            scores[np.array(previous) == 1] -= 100
        return ids, scores
    tree = build_tree(0, lower, 4, 2)
    assert [tree.tokens[node] for node in tree.greedy_path] == [0, 1, 11, 111, 1111]


def test_accepts_alternative_and_excludes_its_sibling():
    tree = ProposalTree([10, 20, 21, 30, 31], [-1, 0, 0, 1, 2], [0, 1, 1, 2, 2], [0, 1, 3])
    path, emitted = accepted_path(tree, [21, 30, 31, 40, 99])
    assert path == [0, 2, 4]
    assert emitted == [21, 31, 99]


def test_rejection_commits_root_and_emits_bonus_only():
    tree = build_tree(0, rows, 8, 2)
    assert accepted_path(tree, [999] * len(tree.tokens)) == ([0], [999])


def test_full_depth_acceptance_includes_one_bonus():
    tree = build_tree(0, rows, 8, 2)
    predictions = [tree.tokens[node] * 10 + 1 for node in range(len(tree.tokens))]
    path, emitted = accepted_path(tree, predictions)
    assert path == tree.greedy_path
    assert len(path) == len(emitted) == 9
    assert emitted[-1] not in tree.tokens


@pytest.mark.parametrize("depth,width", [(0, 2), (9, 2), (4, 0), (4, 4)])
def test_invalid_bounds(depth, width):
    with pytest.raises(ValueError):
        build_tree(0, rows, depth, width)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_draft_ties_retain_native_argmax(width):
    def logsumexp(x, axis, keepdims):
        maximum = np.max(x, axis=axis, keepdims=True)
        return maximum + np.log(np.sum(np.exp(x - maximum), axis=axis, keepdims=keepdims))
    mx = SimpleNamespace(
        array=np.array, stack=np.stack, argmax=np.argmax, concatenate=np.concatenate,
        argpartition=np.argpartition, take_along_axis=np.take_along_axis,
        float32=np.float32, logsumexp=logsumexp, eval=lambda *args: None)
    drafter = SimpleNamespace(markov_head=None, to_target_ids=lambda x: x)
    tree = draft_tree(drafter, np.zeros((8, 11), np.float32), 5, mx, width)
    assert [tree.tokens[node] for node in tree.greedy_path[1:]] == [0] * 8
    assert all(len({tree.tokens[node] for node, parent in enumerate(tree.parents) if parent == candidate})
               == sum(parent == candidate for parent in tree.parents)
               for candidate in range(len(tree.tokens)))
