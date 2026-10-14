"""GLM's retained K3 MTP layer with target-verified generation.

The input norms, normalized target/recurrent states, and shared output
norm follow vLLM's DeepSeek V3.2 MTP equations (Apache-2.0):
https://github.com/vllm-project/vllm/tree/main/vllm/models/deepseek_v32/nvidia
All output tokens are selected by the full target model.
"""

from dataclasses import replace
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import GenerationResponse, generate_step, generation_stream, wired_limit
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList, KVCache, trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .dense import DenseStore
from .model import DecoderLayer


class MTPState(CacheList):
    """Target hidden/token history and the MTP layer's two MLA caches."""

    def __init__(self, *caches):
        super().__init__(*(caches or (KVCache(), KVCache(), KVCache())))

    @property
    def history(self):
        return self.caches[0]

    @property
    def draft(self):
        return CacheList(*self.caches[1:])


class MTPRuntime(nn.Module):
    """Share the target vocabulary and keep a lossless target-state cache."""

    def __init__(self, target, model_path, expert_store, optimize=True):
        super().__init__()
        self.target = target
        self.args = target.args
        self.model_type = target.model_type
        weights = DenseStore(model_path, optimize=optimize).layer(self.args.num_hidden_layers)
        args = replace(
            self.args, num_hidden_layers=self.args.num_hidden_layers + 1,
            indexer_types=[*self.args.indexer_types, "full"],
            mlp_layer_types=[*self.args.mlp_layer_types, "sparse"],
        )
        self.decoder = DecoderLayer(
            args, self.args.num_hidden_layers, weights,
            expert_store.layer(self.args.num_hidden_layers),
        )
        self.eh_proj = weights["eh_proj"]
        self.enorm = weights["enorm.weight"]
        self.hnorm = weights["hnorm.weight"]
        self.output_norm = weights["shared_head.norm.weight"]
        self.freeze()

    @property
    def layers(self):
        return self.target.layers

    def make_cache(self):
        return [*self.target.make_cache(), MTPState()]

    def forward_hidden(self, inputs, cache):
        hidden, indices = self.target.forward_hidden(inputs, cache[:-1])
        normalized = mx.fast.rms_norm(hidden, self.target.norm, self.args.rms_norm_eps)
        cache[-1].history.update_and_fetch(
            normalized[:, None, :, :], inputs[:, None, :, None].astype(mx.uint32),
        )
        return hidden, indices

    def __call__(self, inputs, cache=None):
        if cache is None:
            return self.target(inputs)
        hidden, _ = self.forward_hidden(inputs, cache)
        return self.target.project_hidden(hidden)

    def _prepare(self, inputs, previous_hidden, offset):
        embeddings = self.target.embed_tokens(inputs)
        if offset == 0:
            initial = mx.arange(inputs.shape[1])[None, :, None] == 0
            embeddings = mx.where(initial, 0, embeddings)
        eps = self.args.rms_norm_eps
        return self.eh_proj(mx.concatenate([
            mx.fast.rms_norm(embeddings, self.enorm, eps),
            mx.fast.rms_norm(previous_hidden, self.hnorm, eps),
        ], axis=-1))

    def draft_step(self, inputs, previous_hidden, cache):
        fused = self._prepare(inputs, previous_hidden, cache[0].offset)
        mask = create_attention_mask(fused, cache[0], return_array=True)
        hidden, _ = self.decoder(fused, mask, cache)
        normalized = mx.fast.rms_norm(hidden, self.output_norm, self.args.rms_norm_eps)
        return self.target.lm_head(normalized), normalized

    def sync_draft_cache(self, state, step_size):
        """Pair h[i] with token[i+1] at draft position i.

        vLLM's GLM/DeepSeek MTP proposer shifts token IDs left and keeps
        target positions and states. The final target row waits for its
        next token, so a settled draft cache has history.offset - 1 rows.
        """
        history, draft = state.history, state.draft
        while draft[0].offset < history.offset - 1:
            start = draft[0].offset
            end = min(start + step_size, history.offset - 1)
            inputs = history.values[:, 0, start + 1:end + 1, 0]
            previous = history.keys[:, 0, start:end, :]
            fused = self._prepare(inputs, previous, start)
            norm = mx.fast.rms_norm(fused, self.decoder.input_norm, self.args.rms_norm_eps)
            mask = create_attention_mask(norm, draft[0], return_array=True)
            # Only K/V and indexer keys are needed. Lazy evaluation drops the
            # query, output projection, and expert graphs from this call.
            self.decoder.self_attn(norm, mask, draft)
            mx.eval(draft.state)

    def rebuild_draft_cache(self, state, step_size=128):
        """Replace old draft keys from saved target history; keep history."""
        state.caches = (state.history, KVCache(), KVCache())
        self.sync_draft_cache(state, step_size)


def stream_generate(
    model, tokenizer, prompt, *, max_tokens=128, sampler=None,
    prompt_cache=None, prefill_step_size=128, num_draft_tokens=3,
    prompt_progress_callback=None, stream=generation_stream,
    **kwargs,
):
    """Generate with verified MTP proposals; preserve the target sampler."""
    from mlx_lm import stream_generate as ordinary_generate

    if (not isinstance(model, MTPRuntime) or num_draft_tokens <= 0
            or kwargs.get("draft_model") is not None
            or kwargs.get("logits_processors") or kwargs.get("kv_bits") is not None):
        yield from ordinary_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler,
            prompt_cache=prompt_cache, prefill_step_size=prefill_step_size,
            prompt_progress_callback=prompt_progress_callback, stream=stream,
            num_draft_tokens=num_draft_tokens,
            **kwargs,
        )
        return
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    tokens = mx.array(prompt, dtype=mx.uint32).reshape(-1)
    if tokens.size == 0:
        raise ValueError("The prompt must contain at least one token")
    if max_tokens == 0:
        raise ValueError("Maximum number of tokens must be non-zero")
    if prefill_step_size < 1:
        raise ValueError("The prefill step size must be positive")
    sampler = sampler or (lambda values: mx.argmax(values, axis=-1))
    progress = prompt_progress_callback or (lambda processed, total: None)
    cache = prompt_cache if prompt_cache is not None else model.make_cache()
    state = cache[-1]
    initial_offset = cache[0][0].offset
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    count = 0
    accepted_total = proposed_total = cycles = 0
    proposed_by_depth = [0] * num_draft_tokens
    accepted_by_depth = [0] * num_draft_tokens
    draft_seconds = verify_seconds = sync_seconds = 0.0
    direct_seconds = direct_passes = draft_cost = draft_tokens_checked = 0
    measured_draft_passes = 0
    ordinary_tokens = 0
    ordinary_used = False
    model._mtp_stats = {}

    def sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(logprobs), logprobs

    def record_stats():
        model._mtp_stats = {
            "accepted_draft_tokens": accepted_total,
            "proposed_draft_tokens": proposed_total,
            "verification_cycles": cycles,
            "one_token_passes": direct_passes,
            "ordinary_decode_tokens": ordinary_tokens,
            "measured_draft_passes": measured_draft_passes,
            "adaptive_drafting": True,
            "draft_acceptance": accepted_total / max(proposed_total, 1),
            "acceptance_by_depth": [
                hits / max(proposals, 1)
                for hits, proposals in zip(accepted_by_depth, proposed_by_depth)
            ],
            "draft_seconds": draft_seconds,
            "verify_seconds": verify_seconds,
            "draft_cache_sync_seconds": sync_seconds,
        }

    def emit(token, logprobs, from_draft):
        nonlocal count, finish_reason
        count += 1
        if token in tokenizer.eos_token_ids:
            finish_reason = "stop"
        else:
            detokenizer.add_token(token)
            if count == max_tokens:
                finish_reason = "length"
        if finish_reason is not None:
            detokenizer.finalize()
        return GenerationResponse(
            text=detokenizer.last_segment, token=token, logprobs=logprobs,
            from_draft=from_draft, prompt_tokens=tokens.size, prompt_tps=prompt_tps,
            generation_tokens=count,
            generation_tps=max(count - 1, 0) / max(time.perf_counter() - decode_start, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9, finish_reason=finish_reason,
        )

    with wired_limit(model, [stream]), mx.stream(stream):
        start = time.perf_counter()
        progress(0, tokens.size)
        # Leave the final prompt token for the production decode path.
        for pos in range(0, tokens.size - 1, prefill_step_size):
            part = tokens[pos:min(pos + prefill_step_size, tokens.size - 1)]
            hidden, _ = model.forward_hidden(part[None], cache)
            mx.eval(hidden, [c.state for c in cache[:-1]])
            model.sync_draft_cache(state, prefill_step_size)
            progress(pos + part.size, tokens.size)
            mx.clear_cache()
        hidden, _ = model.forward_hidden(tokens[-1:][None], cache)
        model.sync_draft_cache(state, prefill_step_size)
        seed, seed_logprobs = sample(model.target.project_hidden(hidden)[:, -1, :])
        mx.eval(seed, seed_logprobs)
        progress(tokens.size, tokens.size)
        previous_hidden = state.history.keys[:, 0, state.history.offset - 1:state.history.offset, :]
        prompt_seconds = time.perf_counter() - start
        prompt_tps = tokens.size / prompt_seconds
        decode_start = time.perf_counter()
        pending = [(int(seed.item()), seed_logprobs[0], False)]
        finish_reason = None
        try:
            while max_tokens < 0 or count < max_tokens:
                for token, logprobs, from_draft in pending:
                    yield emit(token, logprobs, from_draft)
                    if finish_reason is not None:
                        return

                # Measure the ordinary pipeline, then use it for the remaining
                # tokens if drafting costs more per committed token.
                if direct_passes == 0 or (measured_draft_passes
                        and draft_cost / draft_tokens_checked >= direct_seconds / direct_passes):
                    probe = direct_passes == 0
                    remaining = -1 if max_tokens < 0 else max_tokens - count
                    limit = min(8, remaining) if probe and remaining >= 0 else (8 if probe else remaining)
                    before = cache[0][0].offset
                    direct_start = time.perf_counter()
                    ordinary_used = True
                    ordinary = generate_step(
                        mx.array([pending[-1][0]], mx.uint32), model, stream=stream,
                        max_tokens=limit, sampler=sampler, prompt_cache=cache,
                        prefill_step_size=prefill_step_size,
                    )
                    try:
                        for token, logprobs in ordinary:
                            ordinary_tokens += 1
                            yield emit(token, logprobs, False)
                            if finish_reason is not None:
                                return
                    finally:
                        ordinary.close()
                        mx.eval([c.state for c in cache])
                        direct_seconds += time.perf_counter() - direct_start
                        direct_passes += cache[0][0].offset - before
                        record_stats()
                    if not probe:
                        return
                    # generate_step has already cached its final emitted token.
                    # Leave that token pending for the next verified pass.
                    trim_prompt_cache(cache, 1)
                    model.sync_draft_cache(state, prefill_step_size)
                    previous_hidden = state.history.keys[:, 0, state.history.offset - 1:state.history.offset, :]
                    pending = [(token, logprobs, False)]

                # The seed has already been emitted, but is not in the target
                # cache yet. Every proposal is checked by the full target.
                draft_offset = state.draft[0].offset
                draft_tokens = []
                draft_start = time.perf_counter()
                y = mx.array([[pending[-1][0]]], dtype=mx.uint32)
                draft_hidden = previous_hidden
                n_draft = num_draft_tokens if max_tokens < 0 else min(num_draft_tokens, max_tokens - count)
                for _ in range(n_draft):
                    draft_logits, draft_hidden = model.draft_step(y, draft_hidden, state.draft)
                    y, _ = sample(draft_logits[:, -1, :])
                    mx.eval(y)
                    draft_tokens.append(int(y.item()))
                    y = y.reshape(1, 1)
                draft_seconds += time.perf_counter() - draft_start
                proposed_total += n_draft
                for depth in range(n_draft):
                    proposed_by_depth[depth] += 1
                verify_start = time.perf_counter()
                verify_inputs = mx.array([[pending[-1][0], *draft_tokens]], dtype=mx.uint32)
                verified_hidden, _ = model.forward_hidden(verify_inputs, cache)
                predictions, logprobs = sample(model.target.project_hidden(verified_hidden)[0])
                mx.eval(predictions, logprobs, verified_hidden)
                verify_seconds += time.perf_counter() - verify_start
                predictions = predictions.tolist()
                accepted = 0
                while accepted < n_draft and predictions[accepted] == draft_tokens[accepted]:
                    accepted += 1
                accepted_total += accepted
                for depth in range(accepted):
                    accepted_by_depth[depth] += 1
                cycles += 1
                rejected = n_draft - accepted
                if rejected:
                    trim_prompt_cache(cache[:-1], rejected)
                    state.history.trim(rejected)
                sync_start = time.perf_counter()
                state.draft.trim(state.draft[0].offset - draft_offset)
                model.sync_draft_cache(state, prefill_step_size)
                sync_end = time.perf_counter()
                sync_seconds += sync_end - sync_start
                pass_seconds = sync_end - draft_start
                if n_draft == num_draft_tokens:
                    draft_cost += pass_seconds
                    draft_tokens_checked += accepted + 1
                    measured_draft_passes += 1
                previous_hidden = state.history.keys[:, 0, state.history.offset - 1:state.history.offset, :]
                pending = [
                    (draft_tokens[i], logprobs[i], True) for i in range(accepted)
                ] + [(int(predictions[accepted]), logprobs[accepted], False)]
                record_stats()
        finally:
            # The HTTP cache key includes every emitted token. A client can
            # stop within a verification pass, or after an uncached bonus.
            finalize_start = time.perf_counter()
            wanted = initial_offset + tokens.size + count
            extra = cache[0][0].offset - wanted
            if extra > 0:
                trim_prompt_cache(cache, extra)
            elif extra < 0:
                model.forward_hidden(mx.array([[token]], dtype=mx.uint32), cache)
                mx.eval([c.state for c in cache[:-1]], state.history.state)
            if ordinary_used or extra < 0:
                model.sync_draft_cache(state, prefill_step_size)
            record_stats()
            model._mtp_stats["cache_finalize_seconds"] = time.perf_counter() - finalize_start
