import mlx.core as mx
import numpy as np
import pytest

from exlr8 import mtp
from test_mtp import DraftRuntime, assert_cache, expected_tokens, tokenizer
from test_mtp_posterior import NoLookup


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("vocabulary", [15, 257, 154880])
def test_decisions_match_independent_posterior_rule(count, vocabulary):
    rng = np.random.default_rng(3030)
    values = rng.normal(size=(count + 1, vocabulary)).astype(np.float32)
    values[0, 0] = 12
    values[1, 1] = -30
    values[:, -1] = -np.inf
    tokens = np.arange(count, dtype=np.uint32)
    probabilities = np.exp(values[:count].astype(np.float64) - values[:count].max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    entropy = -(probabilities * np.log(probabilities + 1e-5)).sum(axis=-1)
    wanted = probabilities[np.arange(count), tokens] > np.minimum(.03, .1 * np.exp(-entropy))
    predictions = np.argmax(values, axis=-1).astype(np.uint32)
    result = mtp._verification_decisions(mx.array(predictions), mx.array(values), mx.array(tokens))
    np.testing.assert_array_equal(np.asarray(result), np.concatenate([predictions, wanted.astype(np.uint32)]))


@pytest.mark.parametrize("cap", [1, 2, 3])
@pytest.mark.parametrize("errors", [(), (0,), (1,), (2,), (0, 1, 2)])
def test_verified_decisions_share_evaluation_and_keep_cache(monkeypatch, cap, errors):
    model = DraftRuntime(errors)
    cache = model.make_cache()
    decision_arrays, evaluated = [], []
    original_decisions, original_eval = mtp._verification_decisions, mx.eval

    def observed_decisions(predictions, logprobs, tokens):
        result = original_decisions(predictions, logprobs, tokens)
        assert result.size == 2 * tokens.size + 1
        decision_arrays.append(result)
        return result

    def observed_eval(*args, **kwargs):
        for result in decision_arrays:
            if any(value is result for value in args):
                assert len(args) == 3
                evaluated.append(result)
        return original_eval(*args, **kwargs)

    def old_host_path(*args):
        pytest.fail("Verification must stage acceptance with predictions.")

    monkeypatch.setattr(mtp, "PromptLookup", NoLookup)
    monkeypatch.setattr(mtp, "_verification_decisions", observed_decisions)
    monkeypatch.setattr(mtp, "_neural_allowed", old_host_path)
    monkeypatch.setattr(mx, "eval", observed_eval)
    responses = list(mtp.stream_generate(model, tokenizer(), [1, 2], max_tokens=cap + 2,
        prompt_cache=cache, num_draft_tokens=cap))
    actual = [response.token for response in responses]
    assert actual == expected_tokens([1, 2], cap + 2)
    assert decision_arrays and len(evaluated) == len(decision_arrays)
    assert_cache(cache, [1, 2, *actual])
