"""Check causal source proposals and streaming target verification."""

import mlx.core as mx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exlr8.mtp import stream_generate
from exlr8.prompt_lookup import PromptLookup
from test_mtp import DraftRuntime, assert_cache


def test_lookup_uses_only_observed_tokens_and_honors_limit():
    lookup = PromptLookup([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5])
    observed = list(lookup.tokens)
    assert lookup.propose(3) == [6, 7, 8]
    assert lookup.tokens == observed
    assert lookup.propose(0) == []
    lookup.append(20)
    assert lookup.cursor is None
    assert lookup.propose(63) == []


def test_lookup_keeps_source_position_until_a_mismatch():
    lookup = PromptLookup([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5])
    assert lookup.propose(3) == [6, 7, 8]
    for token in (6, 7, 8):
        lookup.append(token)
    assert lookup.propose(2) == [9, 10]
    lookup.append(20)
    assert lookup.cursor is None and lookup.propose(2) == []


def cycle_model(monkeypatch):
    import exlr8.mtp as mtp

    model = DraftRuntime((0, 1, 2))
    model.target.project_hidden = lambda hidden: mx.where(
        mx.arange(30) == ((hidden[..., 0] + 1).astype(mx.uint32) % 30)[..., None],
        0.0, -100.0,
    )
    clock = [0.0]
    monkeypatch.setattr(mtp.time, "perf_counter", lambda: clock[0])
    forward = model.target.forward_hidden

    def target(inputs, cache):
        clock[0] += 1.0
        return forward(inputs, cache)

    model.target.forward_hidden = target
    return model


def cycle_tokenizer(eos=29):
    words = {f"t{i}": i for i in range(30)}
    fast = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(words, unk_token="t0")),
        unk_token="t0", eos_token="t29",
    )
    wrapped = TokenizerWrapper(fast)
    wrapped.eos_token_ids = {eos}
    return wrapped


@pytest.mark.parametrize("cap", [1, 2, 3, 7, 15, 63])
@pytest.mark.parametrize("ending", ["length", "eos", "close"])
def test_lookup_preserves_output_stream_and_cache(monkeypatch, cap, ending):
    model = cycle_model(monkeypatch)

    def forbidden(*args):
        raise AssertionError("A matching source span must not run the neural draft layer")

    model.draft_step = forbidden
    cache = model.make_cache()
    prompt = [*range(30), 0, 1, 2, 3]
    rows = []
    forward = model.forward_hidden

    def record(inputs, cache):
        if cache[0][0].offset >= len(prompt):
            rows.append(inputs.shape[1])
        return forward(inputs, cache)

    model.forward_hidden = record
    tok = cycle_tokenizer(eos=18 if ending == "eos" else 99)
    generator = stream_generate(model, tok, prompt, max_tokens=80,
        prompt_cache=cache, num_draft_tokens=cap)
    if ending == "close":
        responses = [next(generator) for _ in range(15)]
        generator.close()
    else:
        responses = list(generator)
    actual = [r.token for r in responses]
    assert actual == [(4 + i) % 30 for i in range(len(actual))]
    assert [r.generation_tokens for r in responses] == list(range(1, len(actual) + 1))
    assert all(r.prompt_tokens == len(prompt) for r in responses)
    assert "".join(r.text for r in responses) == tok.decode(actual[:-1] if ending == "eos" else actual)
    assert_cache(cache, [*prompt, *actual])
    assert model._mtp_stats["lookup_accepted_tokens"] > 0
    assert model._mtp_stats["lookup_proposed_tokens"] == model._mtp_stats["lookup_accepted_tokens"]
    assert model._mtp_stats["ordinary_decode_tokens"] == 0
    assert all(1 <= size <= cap + 1 for size in rows)
    if ending == "eos":
        assert actual[-1] == 18 and responses[-1].finish_reason == "stop"
    elif ending == "length":
        assert len(actual) == 80 and responses[-1].finish_reason == "length"


def test_lookup_rejects_wrong_source_tokens_and_rewinds_cache(monkeypatch):
    model = cycle_model(monkeypatch)
    prompt = [*range(30), 0, 1]
    prompt[16] = 25
    cache = model.make_cache()
    responses = list(stream_generate(model, cycle_tokenizer(eos=99), prompt,
        max_tokens=80, prompt_cache=cache, num_draft_tokens=63))
    actual = [r.token for r in responses]
    assert actual == [(2 + i) % 30 for i in range(80)]
    assert_cache(cache, [*prompt, *actual])
    stats = model._mtp_stats
    assert 0 < stats["lookup_accepted_tokens"] < stats["lookup_proposed_tokens"]


def test_lookup_uses_cached_history_for_a_new_request(monkeypatch):
    model = cycle_model(monkeypatch)
    cache = model.make_cache()
    prompt = [*range(30), 0, 1]
    first = list(stream_generate(model, cycle_tokenizer(eos=99), prompt,
        max_tokens=12, prompt_cache=cache, num_draft_tokens=63))
    first_tokens = [r.token for r in first]

    def forbidden(*args):
        raise AssertionError("Prompt lookup must include the cached token history")

    model.draft_step = forbidden
    second = list(stream_generate(model, cycle_tokenizer(eos=99), [14, 15],
        max_tokens=20, prompt_cache=cache, num_draft_tokens=63))
    second_tokens = [r.token for r in second]
    assert second_tokens == [(16 + i) % 30 for i in range(20)]
    assert model._mtp_stats["lookup_accepted_tokens"] > 0
    assert_cache(cache, [*prompt, *first_tokens, 14, 15, *second_tokens])


@pytest.mark.parametrize("wrong", [False, True])
def test_lookup_checks_a_short_span_before_a_long_copy(monkeypatch, wrong):
    model = cycle_model(monkeypatch)
    limits = []
    propose = PromptLookup.propose

    def lookup(self, limit):
        limits.append(limit)
        return propose(self, limit)

    monkeypatch.setattr(PromptLookup, "propose", lookup)
    prompt = [*range(30), 0, 1, 2, 3]
    if wrong:
        prompt[6] = 25
    cache = model.make_cache()
    forward = model.forward_hidden
    rows = []

    def record(inputs, cache):
        rows.append(inputs.shape[1])
        return forward(inputs, cache)

    model.forward_hidden = record
    generator = stream_generate(model, cycle_tokenizer(eos=99), prompt,
        max_tokens=160, prompt_cache=cache, num_draft_tokens=63)
    responses = [next(generator) for _ in range(2)]
    assert model._mtp_stats["lookup_proposed_tokens"] == 3
    assert model._mtp_stats["lookup_accepted_tokens"] == (1 if wrong else 3)
    responses.extend(generator)
    actual = [r.token for r in responses]
    assert actual == [(4 + i) % 30 for i in range(160)]
    assert_cache(cache, [*prompt, *actual])
    windows = [n for n in rows if n > 1]
    assert windows[1] == 4  # The first is prompt prefill.
    if not wrong:
        assert windows[2] == 16
        assert limits.index(3) < limits.index(15) < limits.index(63)


def test_lookup_ramp_resets_after_a_partial_long_match(monkeypatch):
    model = cycle_model(monkeypatch)
    prompt = [*range(30), 0, 1]
    prompt[21] = 25
    limits = []
    propose = PromptLookup.propose

    def lookup(self, limit):
        limits.append(limit)
        return propose(self, limit)

    monkeypatch.setattr(PromptLookup, "propose", lookup)
    cache = model.make_cache()
    responses = list(stream_generate(model, cycle_tokenizer(eos=99), prompt,
        max_tokens=80, prompt_cache=cache, num_draft_tokens=63))
    actual = [r.token for r in responses]
    assert actual == [(2 + i) % 30 for i in range(80)]
    assert_cache(cache, [*prompt, *actual])
    first_long = limits.index(15)
    assert limits[first_long + 1] == 3
    stats = model._mtp_stats
    assert 0 < stats["lookup_accepted_tokens"] < stats["lookup_proposed_tokens"]
