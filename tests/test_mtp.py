"""Check target verification, cache rewind, and GLM MTP normalization."""

import copy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import CacheList, KVCache, LRUPromptCache, trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from exlr8.mtp import MTPRuntime, MTPState, stream_generate


class Target(nn.Module):
    """Each prediction depends on the full accepted prefix."""

    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(hidden_size=2, rms_norm_eps=1e-5)
        self.layers = []

    def make_cache(self):
        return [CacheList(KVCache())]

    def forward_hidden(self, inputs, cache):
        item = cache[0][0]
        previous_sum = item.keys[:, 0, item.offset - 1, 1] if item.offset else mx.zeros(inputs.shape[0])
        totals = mx.cumsum(inputs.astype(mx.float32), axis=-1) + previous_sum[:, None]
        hidden = mx.stack([inputs.astype(mx.float32), totals], axis=-1)
        item.update_and_fetch(hidden[:, None], hidden[:, None])
        return hidden, None

    def project_hidden(self, hidden):
        next_ids = (hidden[..., 0] + 2 * hidden[..., 1] + 1).astype(mx.uint32) % 9
        return mx.where(mx.arange(10) == next_ids[..., None], 0.0, -100.0)


class DraftRuntime(MTPRuntime):
    def __init__(self, errors=()):
        nn.Module.__init__(self)
        self.target = Target()
        self.args = self.target.args
        self._errors = errors
        self._step = 0

    def forward_hidden(self, inputs, cache):
        hidden, _ = self.target.forward_hidden(inputs, cache[:-1])
        cache[-1].history.update_and_fetch(hidden[:, None], inputs[:, None, :, None])
        return hidden, None

    def sync_draft_cache(self, state, step_size):
        for item in state.draft.caches:
            start = item.offset
            if start < state.history.offset - 1:
                hidden = state.history.keys[:, :, start:state.history.offset - 1]
                item.update_and_fetch(hidden, hidden)

    def draft_step(self, inputs, previous_hidden, cache):
        total = previous_hidden[..., 1] + inputs
        hidden = mx.stack([inputs.astype(mx.float32), total], axis=-1)
        for item in cache.caches:
            item.update_and_fetch(hidden[:, None], hidden[:, None])
        logits = self.target.project_hidden(hidden)
        if self._step % 3 in self._errors:
            logits = mx.roll(logits[..., :9], 1, axis=-1)
            logits = mx.concatenate([logits, mx.full(logits.shape[:-1] + (1,), -100.0)], axis=-1)
        self._step += 1
        return logits, hidden


def tokenizer(eos=9):
    words = {"<unk>": 0, **{f"t{i}": i for i in range(1, 9)}, "<eos>": 9}
    fast = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(words, unk_token="<unk>")),
        unk_token="<unk>", eos_token="<eos>",
    )
    result = TokenizerWrapper(fast)
    result.eos_token_ids = {eos}
    return result


def expected_tokens(prompt, count):
    prefix = list(prompt)
    for _ in range(count):
        prefix.append((prefix[-1] + 2 * sum(prefix) + 1) % 9)
    return prefix[len(prompt):]


def assert_cache(cache, tokens):
    for group in cache[:-1]:
        for item in group.caches:
            assert item.offset == len(tokens)
    assert cache[-1].history.offset == len(tokens)
    for item in cache[-1].draft.caches:
        assert item.offset == max(len(tokens) - 1, 0)
    np.testing.assert_array_equal(np.asarray(cache[-1].history.values)[0, 0, :len(tokens), 0], tokens)
    np.testing.assert_array_equal(np.asarray(cache[0][0].keys)[0, 0, :len(tokens), 0], tokens)


@pytest.mark.parametrize("errors", [(), (0,), (1,), (2,), (0, 1, 2)])
@pytest.mark.parametrize("limit", [1, 2, 4, 11])
@pytest.mark.parametrize("prompt", [[1], [1, 2]])
def test_verification_rewinds_rejected_tokens(errors, limit, prompt):
    model = DraftRuntime(errors)
    cache = model.make_cache()
    responses = list(stream_generate(model, tokenizer(), prompt, max_tokens=limit,
                                     prompt_cache=cache, num_draft_tokens=3, prefill_step_size=1))
    actual = [response.token for response in responses]
    assert actual == expected_tokens(prompt, limit)
    assert responses[-1].finish_reason == "length"
    assert_cache(cache, [*prompt, *actual])
    if not errors and model._mtp_stats.get("proposed_draft_tokens", 0):
        assert model._mtp_stats["draft_acceptance"] == 1


@pytest.mark.parametrize("cap", [1, 3])
@pytest.mark.parametrize("errors", [(), (0,), (1,), (0, 1, 2)])
def test_mtp_uses_production_prefill_and_rewinds_rejected_tokens(cap, errors):
    model = DraftRuntime(errors)
    cache = model.make_cache()
    calls = []
    forward = model.target.forward_hidden

    def tracked(inputs, target_cache):
        calls.append(inputs.shape[1])
        return forward(inputs, target_cache)

    model.target.forward_hidden = tracked
    prompt = [1, 2, 3, 4, 5, 6]
    responses = list(stream_generate(model, tokenizer(), prompt, max_tokens=11,
        prompt_cache=cache, num_draft_tokens=cap, prefill_step_size=2))
    actual = [response.token for response in responses]
    assert actual == expected_tokens(prompt, 11)
    assert calls[:4] == [2, 2, 1, 1]
    assert all(1 <= length <= cap + 1 for length in calls[4:])
    assert_cache(cache, [*prompt, *actual])


def test_eos_and_early_client_stop_keep_cache_keys_aligned():
    model = DraftRuntime()
    cache = model.make_cache()
    responses = list(stream_generate(model, tokenizer(eos=6), [1, 2], max_tokens=-1, prompt_cache=cache))
    assert [response.token for response in responses] == [0, 7, 1, 6]
    assert responses[-1].finish_reason == "stop"
    assert_cache(cache, [1, 2, 0, 7, 1, 6])

    cache = model.make_cache()
    generator = stream_generate(model, tokenizer(), [1, 2], max_tokens=20, prompt_cache=cache)
    outputs = [next(generator).token for _ in range(3)]
    generator.close()
    assert_cache(cache, [1, 2, *outputs])


def test_http_prefix_cache_copy_trim_and_reuse():
    model = DraftRuntime((1,))
    cache = model.make_cache()
    first = list(stream_generate(model, tokenizer(), [1, 2], max_tokens=8, prompt_cache=cache))
    key = [1, 2, *[response.token for response in first]]
    lru = LRUPromptCache(max_size=2)
    lru.insert_cache("mtp", key, cache)
    prompt = [*key[:5], 4, 6]
    restored, rest = lru.fetch_nearest_cache("mtp", prompt)
    assert_cache(restored, prompt[:len(prompt) - len(rest)])
    second = list(stream_generate(model, tokenizer(), rest, max_tokens=6, prompt_cache=restored))
    assert [response.token for response in second] == expected_tokens(prompt, 6)
    assert_cache(cache, key)
    assert_cache(restored, [*prompt, *[response.token for response in second]])
    copied = copy.deepcopy(restored)
    trim_prompt_cache(copied, 2)
    assert copied[-1].history.offset == restored[-1].history.offset - 2


def test_mtp_uses_normalized_target_and_recurrent_states():
    model = MTPRuntime.__new__(MTPRuntime)
    nn.Module.__init__(model)
    model.target = Target()
    model.target.norm = mx.array([0.7, 1.3])
    model.target.embed_tokens = lambda ids: mx.stack([ids, ids + 1], axis=-1).astype(mx.float32)
    model.target.lm_head = lambda hidden: hidden * 2
    model.draft_head = lambda hidden: hidden * 3
    model.args = model.target.args
    model.enorm = mx.array([1.1, 0.8])
    model.hnorm = mx.array([0.6, 1.4])
    model.output_norm = mx.array([1.7, 0.9])
    model.eh_proj = lambda values: values[..., :2] + 0.3 * values[..., 2:]
    model.decoder = lambda values, mask, cache: (values + 0.2, None)
    cache = model.make_cache()
    raw, _ = model.forward_hidden(mx.array([[1, 2]]), cache)
    expected = mx.fast.rms_norm(raw, model.target.norm, model.args.rms_norm_eps)
    np.testing.assert_allclose(np.asarray(cache[-1].history.keys)[0, 0, :2], np.asarray(expected)[0])
    draft = CacheList(KVCache(), KVCache())
    prepared = model._prepare(mx.array([[3]]), expected[:, -1:], 0)
    logits, recurrent = model.draft_step(mx.array([[3]]), expected[:, -1:], draft)
    expected_recurrent = mx.fast.rms_norm(prepared + 0.2, model.output_norm, model.args.rms_norm_eps)
    np.testing.assert_allclose(np.asarray(recurrent), np.asarray(expected_recurrent))
    np.testing.assert_allclose(np.asarray(logits), np.asarray(expected_recurrent * 3))
    np.testing.assert_allclose(np.asarray(model.target.lm_head(recurrent)), np.asarray(expected_recurrent * 2))


def test_first_teacher_row_keeps_h0_and_masks_only_position_zero_embedding():
    model = MTPRuntime.__new__(MTPRuntime)
    nn.Module.__init__(model)
    model.target = Target()
    model.args = model.target.args
    model.enorm = mx.array([1.1, 0.8])
    model.hnorm = mx.array([0.6, 1.4])
    seen_ids, fused_inputs = [], []

    def embed(ids):
        seen_ids.append(ids)
        return mx.stack([ids, ids + 1], axis=-1).astype(mx.float32)

    def project(values):
        fused_inputs.append(values)
        return values[..., :2] + values[..., 2:]

    def attention(values, mask, cache):
        for item in cache.caches:
            item.update_and_fetch(values[:, None], values[:, None])

    model.target.embed_tokens = embed
    model.eh_proj = project
    model.decoder = SimpleNamespace(input_norm=mx.ones(2), self_attn=attention)
    hidden = mx.array([[[1.0, 2.0], [4.0, -2.0], [9.0, 10.0]]])
    state = MTPState()
    state.history.update_and_fetch(hidden[:, None], mx.array([[[[6], [7], [3]]]]))
    model.sync_draft_cache(state, step_size=2)

    np.testing.assert_array_equal(np.asarray(seen_ids[0]), [[7, 3]])
    expected_h = mx.fast.rms_norm(hidden[:, :2], model.hnorm, model.args.rms_norm_eps)
    expected_e1 = mx.fast.rms_norm(mx.array([[3.0, 4.0]]), model.enorm, model.args.rms_norm_eps)
    np.testing.assert_array_equal(np.asarray(fused_inputs[0][0, 0, :2]), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(fused_inputs[0][..., 2:]), np.asarray(expected_h))
    np.testing.assert_allclose(np.asarray(fused_inputs[0][0, 1, :2]), np.asarray(expected_e1[0]))
    assert state.draft[0].offset == state.draft[1].offset == 2


def test_teacher_forced_mtp_cache_matches_full_layer(monkeypatch):
    from test_model import fixtures, Linear
    from exlr8.model import DecoderLayer, Model

    args, weights, experts, vocab = fixtures(topk=3)
    weights = dict(weights, **{
        "eh_proj": Linear(np.random.default_rng(6).normal(0, 0.1, (8, 16)).astype(np.float32)),
        "enorm.weight": mx.linspace(0.7, 1.3, 8),
        "hnorm.weight": mx.linspace(1.4, 0.6, 8),
        "shared_head.norm.weight": mx.linspace(0.9, 1.2, 8),
    })
    vocab["norm.weight"] = mx.linspace(0.6, 1.4, 8)
    monkeypatch.setattr("exlr8.mtp.DenseStore", lambda *a, **kw: SimpleNamespace(layer=lambda index: weights))
    target = Model(args, [DecoderLayer(args, 0, weights)], vocab)
    model = MTPRuntime(target, "unused", SimpleNamespace(layer=lambda index: experts))
    cache = model.make_cache()
    inputs = mx.array([[1, 2, 3, 4, 5]])
    model.forward_hidden(inputs, cache)
    model.sync_draft_cache(cache[-1], 2)
    normalized = cache[-1].history.keys[:, 0, :5]
    complete = CacheList(KVCache(), KVCache())
    fused = model._prepare(inputs[:, 1:], normalized[:, :-1], 0)
    mask = mx.arange(4)[:, None] >= mx.arange(4)[None, :]
    model.decoder(fused, mask, complete)
    for actual, expected in zip(cache[-1].draft.caches, complete.caches):
        assert actual.offset == expected.offset == 4
        np.testing.assert_allclose(np.asarray(actual.keys)[:, :, :4], np.asarray(expected.keys)[:, :, :4], rtol=2e-5, atol=2e-6)

    seed = mx.array([[6]])
    expected_logits, expected_hidden = model.draft_step(seed, normalized[:, -1:], complete)
    logits, hidden = model.draft_step(seed, normalized[:, -1:], cache[-1].draft)
    np.testing.assert_allclose(np.asarray(logits), np.asarray(expected_logits), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(hidden), np.asarray(expected_hidden), rtol=2e-5, atol=2e-6)
    assert cache[0][0].offset == cache[-1].history.offset == 5
    assert cache[-1].draft[0].offset == 5

    # A saved prefix can contain the old N-row convention. Rebuild only
    # the copied draft keys and preserve every target/history array.
    copied = copy.deepcopy(cache)
    target_keys = np.array(copied[0][0].keys)
    history = copied[-1].history
    history_keys = np.array(history.keys)
    history_values = np.array(history.values)
    model.rebuild_draft_cache(copied[-1], step_size=2)
    assert copied[-1].history is history
    assert copied[0][0].offset == history.offset == 5
    assert copied[-1].draft[0].offset == copied[-1].draft[1].offset == 4
    np.testing.assert_array_equal(np.asarray(copied[0][0].keys), target_keys)
    np.testing.assert_array_equal(np.asarray(history.keys), history_keys)
    np.testing.assert_array_equal(np.asarray(history.values), history_values)
    for actual, expected in zip(copied[-1].draft.caches, complete.caches):
        np.testing.assert_allclose(np.asarray(actual.keys)[:, :, :4],
                                   np.asarray(expected.keys)[:, :, :4], rtol=2e-5, atol=2e-6)
    assert cache[-1].draft[0].offset == 5


@pytest.mark.parametrize("cap", [1, 3])
@pytest.mark.parametrize("draft_faster", [False, True])
def test_drafting_uses_measured_cost_and_preserves_target_cache(monkeypatch, cap, draft_faster):
    import exlr8.mtp as mtp_module

    monkeypatch.setattr(mtp_module.PromptLookup, "propose", lambda self, limit: [])
    model = DraftRuntime()
    clock = [0.0]
    monkeypatch.setattr(mtp_module.time, "perf_counter", lambda: clock[0])
    target_forward = model.target.forward_hidden
    draft_step = model.draft_step
    calls = []

    def target(inputs, cache):
        rows = inputs.shape[1]
        calls.append(rows)
        clock[0] += 1.0 if rows == 1 or draft_faster else 2.0 * rows
        return target_forward(inputs, cache)

    def draft(*args):
        clock[0] += 0.1
        return draft_step(*args)

    model.target.forward_hidden = target
    model.draft_step = draft
    cache = model.make_cache()
    prompt = [1, 2]
    output = list(stream_generate(model, tokenizer(), prompt, max_tokens=40,
        prompt_cache=cache, num_draft_tokens=cap))
    tokens = [row.token for row in output]
    assert tokens == expected_tokens(prompt, 40)
    assert_cache(cache, [*prompt, *tokens])
    assert model._mtp_stats["adaptive_drafting"]
    assert model._mtp_stats["seed_target_seconds"] == 1.0
    assert (model._mtp_stats["one_token_passes"] == 0) == draft_faster
    assert model._mtp_stats["measured_draft_passes"] >= 1
    if draft_faster:
        assert model._mtp_stats["measured_draft_passes"] > 1
    else:
        assert model._mtp_stats["measured_draft_passes"] == 4
        wide = [i for i, rows in enumerate(calls) if rows > 1]
        assert len(wide) == 4
        assert all(rows == 1 for rows in calls[wide[-1] + 1:])


@pytest.mark.parametrize("cap", [1, 3, 7])
@pytest.mark.parametrize("ending", ["length", "eos", "close"])
def test_slow_draft_uses_ordinary_pipeline_and_preserves_stream_and_cache(monkeypatch, cap, ending):
    import exlr8.mtp as mtp_module
    from mlx_lm.generate import generate_step

    class NoMatch:
        def __init__(self, tokens):
            pass

        def append(self, token):
            pass

        def propose(self, limit):
            return []

    monkeypatch.setattr(mtp_module, "PromptLookup", NoMatch)
    model = DraftRuntime((0, 1, 2))
    clock = [0.0]
    monkeypatch.setattr(mtp_module.time, "perf_counter", lambda: clock[0])
    forward = model.target.forward_hidden
    draft = model.draft_step

    def target(inputs, cache):
        clock[0] += 1.0 if inputs.shape[1] == 1 else 2.0 * inputs.shape[1]
        return forward(inputs, cache)

    def propose(*args):
        clock[0] += .1
        return draft(*args)

    pipeline_calls = []
    def pipeline(*args, **kwargs):
        pipeline_calls.append(kwargs["max_tokens"])
        yield from generate_step(*args, **kwargs)

    model.target.forward_hidden = target
    model.draft_step = propose
    monkeypatch.setattr(mtp_module, "generate_step", pipeline, raising=False)
    cache = model.make_cache()
    prompt = [1, 2]
    generator = stream_generate(model, tokenizer(eos=6 if ending == "eos" else 9), prompt,
        max_tokens=40, prompt_cache=cache, num_draft_tokens=cap)
    if ending == "close":
        responses = [next(generator) for _ in range(15)]
        generator.close()
    else:
        responses = list(generator)
    tokens = [row.token for row in responses]
    assert tokens == expected_tokens(prompt, len(tokens))
    assert [row.generation_tokens for row in responses] == list(range(1, len(tokens) + 1))
    assert all(row.prompt_tokens == len(prompt) for row in responses)
    assert_cache(cache, [*prompt, *tokens])
    if ending == "length":
        assert len(pipeline_calls) == 1
        assert len(tokens) == 40 and responses[-1].finish_reason == "length"
    elif ending == "eos":
        assert not pipeline_calls
        assert tokens[-1] == 6 and responses[-1].finish_reason == "stop"


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("limit", [1, 2, 20, 80])
def test_seed_timing_excludes_draft_sync_and_starts_drafting_immediately(monkeypatch, cached, limit):
    import exlr8.mtp as mtp_module

    monkeypatch.setattr(mtp_module.PromptLookup, "propose", lambda self, limit: [])
    model = DraftRuntime()
    cache = model.make_cache()
    prefix = [1, 2] if cached else []
    if cached:
        model.forward_hidden(mx.array([prefix], mx.uint32), cache)
        model.sync_draft_cache(cache[-1], 128)
        mx.eval([c.state for c in cache])
    clock = [0.0]
    monkeypatch.setattr(mtp_module.time, "perf_counter", lambda: clock[0])
    forward = model.target.forward_hidden
    sync = model.sync_draft_cache
    calls = []

    def target(inputs, target_cache):
        calls.append(inputs.shape[1])
        clock[0] += 1.0
        return forward(inputs, target_cache)

    def slow_sync(*args):
        clock[0] += 10.0
        return sync(*args)

    model.target.forward_hidden = target
    model.sync_draft_cache = slow_sync
    output = list(stream_generate(model, tokenizer(), [3], max_tokens=limit,
        prompt_cache=cache, num_draft_tokens=3))
    actual = [row.token for row in output]
    assert actual == expected_tokens([*prefix, 3], limit)
    assert model._mtp_stats["seed_target_seconds"] == 1.0
    assert_cache(cache, [*prefix, 3, *actual])
    if limit > 1:
        assert calls[1] == min(3, limit - 1) + 1
    if limit <= 2:
        assert model._mtp_stats["ordinary_decode_seconds"] == 0
        assert model._mtp_stats["one_token_passes"] == 0


@pytest.mark.parametrize("cap", [1, 2, 3])
@pytest.mark.parametrize("errors", [(), (0,), (2,)])
def test_neural_proposals_have_one_final_host_wait(monkeypatch, cap, errors):
    model = DraftRuntime(errors)
    cache = model.make_cache()
    evaluations = 0
    draft_evaluations = []
    original_eval = mx.eval
    original_step = model.draft_step

    def counted_eval(*args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return original_eval(*args, **kwargs)

    def observed_step(*args, **kwargs):
        draft_evaluations.append(evaluations)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", counted_eval)
    model.draft_step = observed_step
    responses = list(stream_generate(model, tokenizer(), [1, 2],
        max_tokens=cap + 1, prompt_cache=cache, num_draft_tokens=cap))
    assert len(draft_evaluations) >= cap
    assert len(set(draft_evaluations[:cap])) == 1
    actual = [response.token for response in responses]
    assert actual == expected_tokens([1, 2], cap + 1)
    assert_cache(cache, [1, 2, *actual])
