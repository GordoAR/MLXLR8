"""CPU-only checks of the private repair adapter and cache control flow."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mtp1_retain_rejection import (
    _NATIVE_REPAIR, _RETAIN_REPAIR, _repair, build_stream_generate,
)


class Cache:
    def __init__(self, offset):
        self.offset = offset
        self.trims = []

    def trim(self, count):
        self.trims.append(count)
        self.offset -= count


class Draft:
    def __init__(self, offset):
        self.caches = [Cache(offset), Cache(offset)]

    def __getitem__(self, index):
        return self.caches[index]

    def trim(self, count):
        for cache in self.caches:
            cache.trim(count)


class Runtime:
    def __init__(self):
        self.syncs = []

    def sync_draft_cache(self, state, step_size):
        self.syncs.append(step_size)
        for cache in state.draft.caches:
            cache.offset = state.history.offset - 1


@pytest.mark.parametrize("cap,proposed,rejected", [
    (1, 1, 1), (1, 1, 0), (2, 1, 1), (3, 3, 3), (3, 3, 1), (0, 0, 0),
])
def test_only_rejected_mtp1_removes_repair(cap, proposed, rejected):
    model = Runtime()
    start = 100
    state = SimpleNamespace(history=Cache(start + 2 + proposed - rejected),
                            draft=Draft(start + proposed))
    _repair(model, state, start, 128, cap, proposed, rejected)
    retain = cap == proposed == rejected == 1
    assert model.syncs == ([] if retain else [128])
    for cache in state.draft.caches:
        assert cache.offset == state.history.offset - 1
        assert cache.trims == ([] if retain else [proposed])


def fake_stream_generate(model, state, outcomes, num_draft_tokens=1, limit=8,
                         stop_after=None, prefill_step_size=128):
    """CPU model of native repair placement and generator finalization."""
    count = 1  # Seed has been emitted but is not yet cached.
    try:
        while count < limit:
            for accepted in outcomes:
                draft_offset = state.draft[0].offset
                n_draft = min(num_draft_tokens, limit - count)
                accepted = min(accepted, n_draft)
                rejected = n_draft - accepted
                for cache in state.draft.caches:
                    cache.offset += n_draft
                state.history.offset += 1 + accepted
                state.draft.trim(state.draft[0].offset - draft_offset)
                model.sync_draft_cache(state, prefill_step_size)
                for _ in range(accepted + 1):
                    count += 1
                    yield count
                    if count == limit or count == stop_after:
                        return
            return
    finally:
        # Match the native finalization contract for a cached accepted row
        # or an emitted, uncached bonus. Initial history length is 100.
        wanted = 100 + count
        extra = state.history.offset - wanted
        if extra > 0:
            state.history.trim(extra)
            state.draft.trim(extra)
        elif extra < 0:
            state.history.offset += 1
            model.sync_draft_cache(state, prefill_step_size)


@pytest.mark.parametrize("outcomes,limit,stop_after,cancel_after", [
    ([0, 0, 0], 4, None, None),  # All rejected; end at an uncached token.
    ([1, 1, 1], 5, None, None),  # All accepted.
    ([0, 1, 0], 5, None, None),
    ([1, 1], 8, 2, None),       # Stop after the accepted token.
    ([0, 1], 8, 2, None),       # Stop after a rejected proposal.
    ([1, 1], 8, None, 1),       # Close while suspended at a yield.
    ([0, 1], 8, None, 1),
    ([0], 1, None, None),       # No proposal; finalization still runs.
])
@pytest.mark.parametrize("cap", [1, 3])
def test_native_and_private_lifecycle(outcomes, limit, stop_after, cancel_after, cap):
    private = build_stream_generate(fake_stream_generate)
    original = SimpleNamespace(history=Cache(100), draft=Draft(99))
    results = []
    for generate in (fake_stream_generate, private):
        model, state = Runtime(), copy.deepcopy(original)
        generator = generate(model, state, outcomes, cap, limit, stop_after)
        emitted = []
        for value in generator:
            emitted.append(value)
            if cancel_after == len(emitted):
                generator.close()
                break
        results.append((emitted, state.history.offset,
                        [c.offset for c in state.draft.caches], model.syncs))
        assert state.history.offset == 101 + len(emitted)
        assert all(c.offset == state.history.offset - 1 for c in state.draft.caches)
    assert results[0][:3] == results[1][:3]
    if cap != 1:
        assert results[0][3] == results[1][3]
    assert original.history.offset == 100
    assert original.draft[0].offset == 99


def test_current_production_source_changes_only_repair():
    path = Path(__file__).resolve().parents[1] / "exlr8" / "mtp.py"
    source = path.read_text()
    assert source.count(_NATIVE_REPAIR) == 1
    changed = source.replace(_NATIVE_REPAIR, _RETAIN_REPAIR)
    compile(changed, str(path), "exec")  # No imports or MLX execution.
    assert changed.replace(_RETAIN_REPAIR, _NATIVE_REPAIR) == source
    original_finally = next(node for node in ast.walk(ast.parse(source))
                            if isinstance(node, ast.Try) and node.finalbody)
    changed_finally = next(node for node in ast.walk(ast.parse(changed))
                           if isinstance(node, ast.Try) and node.finalbody)
    assert ast.dump(ast.Module(body=original_finally.finalbody, type_ignores=[])) == ast.dump(
        ast.Module(body=changed_finally.finalbody, type_ignores=[]))


def test_source_drift_fails_closed():
    with pytest.raises(ValueError, match="repair block changed"):
        build_stream_generate(test_source_drift_fails_closed)
