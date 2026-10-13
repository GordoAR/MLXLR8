"""Private MTP1 cache experiment. Real-cache exactness requires a GPU check."""

import copy
import inspect
import textwrap


_NATIVE_REPAIR = """                state.draft.trim(state.draft[0].offset - draft_offset)
                model.sync_draft_cache(state, prefill_step_size)"""
_RETAIN_REPAIR = """                _mtp1_repair(model, state, draft_offset, prefill_step_size,
                             num_draft_tokens, n_draft, rejected)"""


def _repair(model, state, draft_offset, step_size, cap, proposed, rejected):
    if cap == proposed == rejected == 1:
        return
    state.draft.trim(state.draft[0].offset - draft_offset)
    model.sync_draft_cache(state, step_size)


def build_stream_generate(native=None):
    """Return a private generator; do not change the production module."""
    if native is None:
        from exlr8.mtp import stream_generate as native
    source = textwrap.dedent(inspect.getsource(native))
    if source.count(_NATIVE_REPAIR) != 1:
        raise ValueError("The native draft-cache repair block changed")
    namespace = dict(native.__globals__, _mtp1_repair=_repair)
    exec(compile(source.replace(_NATIVE_REPAIR, _RETAIN_REPAIR),
                 "<mtp1-retain-rejection>", "exec"), namespace)
    return namespace[native.__name__]


def compare_rejected_state(model, state, draft_offset, step_size=128):
    """Compare both repairs on copies of one real rejected MTP1 state.

    Call immediately before native draft repair: target/history rejection
    trimming must be complete, and the single proposed draft row must remain.
    Includes every allocated history/draft key and value, plus all offsets.
    Does not change the supplied state. This is a check, not a timing sample.
    """
    import mlx.core as mx

    expected = draft_offset + 1
    if (state.history.offset != expected + 1
            or any(c.offset != expected for c in state.draft.caches)):
        raise ValueError("Expected a rejected MTP1 state before draft repair")
    native, retained = copy.deepcopy(state), copy.deepcopy(state)
    _repair(model, native, draft_offset, step_size, 0, 1, 1)
    _repair(model, retained, draft_offset, step_size, 1, 1, 1)
    mx.eval(native.state, retained.state)
    rows, checks = [], []
    for index, (left, right) in enumerate(zip(native.caches, retained.caches)):
        rows.append({"cache": index, "offsets_equal": left.offset == right.offset,
                     "native_offset": left.offset, "retained_offset": right.offset})
        for name in ("keys", "values"):
            a, b = getattr(left, name), getattr(right, name)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("Cache shape or dtype differs")
            changed = mx.sum(mx.contiguous(a).view(mx.uint8)
                             != mx.contiguous(b).view(mx.uint8))
            finite = mx.all(mx.isfinite(a)) & mx.all(mx.isfinite(b))
            checks.append((index, name, changed, finite))
    mx.eval([value for _, _, changed, finite in checks for value in (changed, finite)])
    for index, name, changed, finite in checks:
        rows[index][name] = {"changed_bytes": int(changed.item()),
                             "finite": bool(finite.item())}
    exact = all(row["offsets_equal"] and all(
        row[name]["changed_bytes"] == 0 and row[name]["finite"]
        for name in ("keys", "values")) for row in rows)
    return {"exact": exact, "caches": rows}
