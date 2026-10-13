"""Private bounded DSpark proposal trees. No MLX import or target changes."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ProposalTree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]
    greedy_path: list[int]


def build_tree(anchor, candidates, depth=8, width=2):
    """Keep the best width paths at each depth, including the greedy trunk.

    candidates(previous_tokens, position, width) returns target-vocabulary IDs
    and conditional log probabilities, sorted by probability for each parent.
    The DSpark backbone runs once; its Markov head supplies each conditional row.
    """
    if not 1 <= depth <= 8 or not 1 <= width <= 3:
        raise ValueError("Private tree supports depth 1..8 and width 1..3")
    tree = ProposalTree([int(anchor)], [-1], [0], [0])
    frontier, scores = [0], [0.0]
    for position in range(depth):
        ids, logp = candidates([tree.tokens[node] for node in frontier], position, width)
        ids, logp = np.asarray(ids), np.asarray(logp)
        if ids.shape != (len(frontier), width) or logp.shape != ids.shape:
            raise ValueError("Conditional candidate rows have the wrong shape")
        choices = [(scores[row] + float(logp[row, rank]), parent, rank, int(ids[row, rank]))
                   for row, parent in enumerate(frontier) for rank in range(width)]
        choices.sort(key=lambda item: (-item[0], item[1], item[2]))
        trunk_parent = tree.greedy_path[-1]
        trunk = next(item for item in choices if item[1] == trunk_parent and item[2] == 0)
        selected = choices[:width]
        if trunk not in selected:
            selected[-1] = trunk
        next_frontier, next_scores = [], []
        for score, parent, _, token in selected:
            node = len(tree.tokens)
            tree.tokens.append(token)
            tree.parents.append(parent)
            tree.depths.append(position + 1)
            next_frontier.append(node)
            next_scores.append(score)
            if parent == trunk_parent and token == trunk[3]:
                tree.greedy_path.append(node)
        frontier, scores = next_frontier, next_scores
    return tree


def accepted_path(tree, predictions):
    """Return cache rows and new tokens: accepted children, then one bonus.

    The root token was already emitted. predictions[node] is the target's
    greedy next token after that node. The bonus is not put in the target cache.
    """
    if len(predictions) != len(tree.tokens):
        raise ValueError("Each verified node needs one target prediction")
    children = [[] for _ in tree.tokens]
    for node, parent in enumerate(tree.parents[1:], 1):
        children[parent].append(node)
    path, emitted = [0], []
    while True:
        current = path[-1]
        token = int(predictions[current])
        child = next((node for node in children[current] if tree.tokens[node] == token), None)
        emitted.append(token)
        if child is None:
            return path, emitted
        path.append(child)


def draft_tree(drafter, base_logits, anchor, mx, width=2):
    """Use the existing BF16 logits and Markov addition before ranking.

    Per-parent Markov calls keep the width-one draft path identical to the
    public drafter. F32 log probabilities only rank alternative paths.
    """
    def candidates(previous, position, count):
        rows = []
        for token in previous:
            row = base_logits[position]
            if drafter.markov_head is not None:
                row = row + drafter.markov_head.step_bias(mx.array([token]))[0]
            rows.append(row)
        logits = mx.stack(rows)
        # Keep argmax explicitly: an equal-logit partition can omit its tie winner.
        first = mx.argmax(logits, axis=-1, keepdims=True)
        ids = first if count == 1 else mx.concatenate(
            (first, mx.argpartition(-logits, count - 1, axis=-1)[:, :count]), axis=-1)
        scores = mx.take_along_axis(logits.astype(mx.float32), ids, axis=-1)
        logp = scores - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
        target_ids = drafter.to_target_ids(ids)
        mx.eval(ids, target_ids, scores, logp)
        local, target, scores, logp = map(np.asarray, (ids, target_ids, scores, logp))
        selected = []
        for row in range(len(previous)):
            order = sorted(range(local.shape[1]), key=lambda col: (-scores[row, col], local[row, col]))
            unique = []
            for col in order:
                if local[row, col] not in [local[row, old] for old in unique]:
                    unique.append(col)
            selected.append(unique[:count])
        selected = np.array(selected)
        return (np.take_along_axis(target, selected, axis=-1),
                np.take_along_axis(logp, selected, axis=-1))

    return build_tree(anchor, candidates, depth=base_logits.shape[0], width=width)
