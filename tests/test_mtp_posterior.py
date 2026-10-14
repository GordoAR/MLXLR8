"""Relaxed words retain actual history and stop at response boundaries."""

import time

import mlx.core as mx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exlr8 import mtp
from test_mtp import Target, DraftRuntime, tokenizer, assert_cache


class TwoChoiceTarget(Target):
    def project_hidden(self, hidden):
        words = mx.arange(10)
        logits = mx.where(words == 3, 0.0, mx.where(words == 4, -.4, -100.0))
        return mx.broadcast_to(logits, hidden.shape[:-1] + (10,))


class NoLookup:
    def __init__(self, tokens):
        pass

    def append(self, token):
        pass

    def propose(self, limit):
        return []


def prepared(monkeypatch):
    monkeypatch.setattr(mtp, 'PromptLookup', NoLookup)
    original = mtp.generate_step

    def paced(*args, **kwargs):
        for item in original(*args, **kwargs):
            time.sleep(.001)
            yield item

    monkeypatch.setattr(mtp, 'generate_step', paced)
    model = DraftRuntime(errors=(0, 1, 2))
    model.target = TwoChoiceTarget()
    return model


def test_posterior_allows_plausible_word_and_rejects_rare_word():
    with mx.stream(mx.cpu):
        logits = mx.array([[0.0, -2.5, -100.0], [0.0, -8.0, -100.0]])
        assert mtp._neural_allowed(logits, [1, 1]) == [True, False]


@pytest.mark.parametrize('cap', [1, 3, 63])
def test_relaxed_words_keep_actual_history(monkeypatch, cap):
    with mx.stream(mx.cpu):
        model = prepared(monkeypatch)
        cache = model.make_cache()
        replies = list(mtp.stream_generate(model, tokenizer(), [1, 2], max_tokens=90,
            num_draft_tokens=cap, prompt_cache=cache, prefill_step_size=1))
        assert len(replies) == 90 and replies[-1].finish_reason == 'length'
        assert model._mtp_stats['non_greedy_accepted_tokens'] > 0
        assert model._mtp_stats['neural_limit_current'] == min(cap, 3)
        assert not any(model._mtp_stats['acceptance_by_depth'][3:])
        assert_cache(cache, [1, 2, *[r.token for r in replies]])


@pytest.mark.parametrize('stop_after', [10, 12, 17])
def test_client_stop_with_relaxed_words_keeps_actual_history(monkeypatch, stop_after):
    with mx.stream(mx.cpu):
        model = prepared(monkeypatch)
        cache = model.make_cache()
        stream = mtp.stream_generate(model, tokenizer(), [1, 2], max_tokens=90,
            num_draft_tokens=63, prompt_cache=cache, prefill_step_size=1)
        tokens = [next(stream).token for _ in range(stop_after)]
        stream.close()
        assert_cache(cache, [1, 2, *tokens])


@pytest.mark.parametrize('target_stops', [True, False])
def test_stop_transition_uses_the_target_choice(monkeypatch, target_stops):
    class BoundaryTarget(TwoChoiceTarget):
        def project_hidden(self, hidden):
            words = mx.arange(10)
            boundary = mx.where(words == (9 if target_stops else 3), 0.0, -.4)
            return mx.where((hidden[..., 1] >= 60)[..., None], boundary,
                            super().project_hidden(hidden))

    class BoundaryDraft(DraftRuntime):
        def draft_step(self, inputs, previous_hidden, cache):
            logits, hidden = super().draft_step(inputs, previous_hidden, cache)
            words = mx.arange(10)
            wrong = mx.where(words == (1 if target_stops else 9), 0.0, -100.0)
            return mx.where((hidden[..., 1] >= 60)[..., None], wrong, logits), hidden

    with mx.stream(mx.cpu):
        prepared(monkeypatch)
        model = BoundaryDraft()
        model.target = BoundaryTarget()
        cache = model.make_cache()
        replies = list(mtp.stream_generate(model, tokenizer(), [1, 2], max_tokens=90,
            num_draft_tokens=63, prompt_cache=cache, prefill_step_size=1))
        tokens = [r.token for r in replies]
        assert replies[-1].finish_reason == ('stop' if target_stops else 'length')
        assert (tokens[-1] == 9) if target_stops else (9 not in tokens and len(tokens) == 90)
        if target_stops:
            assert sum([1, 2, *tokens[:-2]]) < 60
            assert sum([1, 2, *tokens[:-1]]) >= 60
        assert_cache(cache, [1, 2, *tokens])


def test_assistant_role_marker_ends_this_response(monkeypatch):
    words = {'<unk>': 0, **{f't{i}': i for i in range(1, 9) if i != 4},
             '<|assistant|>': 4, '<eos>': 9}
    fast = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(words, unk_token='<unk>')),
        unk_token='<unk>', eos_token='<eos>', additional_special_tokens=['<|assistant|>'])
    with mx.stream(mx.cpu):
        model = prepared(monkeypatch)
        model._errors = ()
        model.target.project_hidden = lambda h: mx.broadcast_to(
            mx.where(mx.arange(10) == 4, 0.0, -100.0), h.shape[:-1] + (10,))
        cache = model.make_cache()
        replies = list(mtp.stream_generate(model, TokenizerWrapper(fast), [1, 2], max_tokens=90,
            num_draft_tokens=63, prompt_cache=cache, prefill_step_size=1))
        assert len(replies) == 1 and replies[0].token == 4
        assert replies[0].finish_reason == 'stop'
        assert_cache(cache, [1, 2, 4])
