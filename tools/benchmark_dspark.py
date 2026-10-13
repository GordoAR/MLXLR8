"""Test the public GLM DSpark drafter against the resident EXLR8 target."""

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache

from exlr8.cli import DEFAULT_MODEL, load_runtime
from tools.benchmark_runtime import benchmark, load_prefix


def matrix_layers(model, first_layer):
    selected = {}
    if first_layer is None:
        return selected
    for index, layer in enumerate(model.layers):
        expert = layer.experts
        if index < first_layer or expert is None:
            continue
        if expert.k != 4 or not expert.store.resident or not expert.store.optimize:
            raise ValueError(f"Private matrix layer {index} requires optimized resident K4 experts")
        selected[id(expert)] = (index, expert)
    if not selected:
        raise ValueError("--matrix-from-layer selects no target sparse layers")
    return selected


def compiled_expert_calls(selected, matrix):
    from exlr8.experts import ExpertLayer
    from exlr8.fast_prefill import grouped_expert_mlp

    original = ExpertLayer.__call__

    def build(expert):
        def native(x, ids):
            return original(expert, x, ids)

        def grouped(x, ids):
            return grouped_expert_mlp(x, ids, expert.packed, expert.scales,
                                      expert.k, expert.hidden, expert.intermediate)

        return mx.compile(grouped if matrix else native)

    return {identity: build(expert) for identity, (_, expert) in selected.items()}


@contextmanager
def private_expert_calls(selected, calls):
    from exlr8.experts import ExpertLayer

    original = ExpertLayer.__call__
    evidence = Counter()

    def apply(expert, x, ids):
        identity = id(expert)
        rows = ids.size // ids.shape[-1]
        if identity in calls and 5 <= rows <= 16:
            index = selected[identity][0]
            evidence[index, tuple(x.shape), tuple(ids.shape), str(x.dtype)] += 1
            return calls[identity](x, ids)
        return original(expert, x, ids)

    ExpertLayer.__call__ = apply
    try:
        yield evidence
    finally:
        ExpertLayer.__call__ = original


def first_difference(reference, actual):
    return next((i for i, (a, b) in enumerate(zip(reference, actual)) if a != b),
                min(len(reference), len(actual)) if len(reference) != len(actual) else None)


def wrap_target(model, tokenizer):
    from mlx_dspark.target import Target

    class EXLR8Target(Target):
        def _body_mlxlm(self, ids, cache, tap):
            captured = {}
            previous = self.model.trace

            def capture(index, hidden):
                # vLLM captures each configured layer INPUT. Our hook receives
                # the preceding layer's output, before the target final norm.
                if tap is not None and index + 1 in tap:
                    captured[index + 1] = hidden
                if previous is not None:
                    previous(index, hidden)

            self.model.trace = capture if tap is not None else previous
            try:
                hidden, _ = self.model.forward_hidden(ids, cache)
            finally:
                self.model.trace = previous
            fused = mx.concatenate([captured[index] for index in tap], axis=-1) if tap is not None else None
            normalized = mx.fast.rms_norm(hidden, self.model.norm, self.model.args.rms_norm_eps)
            return normalized, fused

        def draft_embed(self, ids):
            return self.model.embed_tokens(ids)

    return EXLR8Target(model, tokenizer)


def prepare_draft_prefix(target, drafter, cache, ids, step_size):
    """Replay only the trained sliding window from the saved target prefix."""
    config = drafter.config
    if not config.causal_block or not config.sliding_window:
        raise ValueError("This prefix shortcut requires the trained causal sliding window")
    end = len(ids) - 1
    start = max(0, ((end - config.sliding_window) // step_size) * step_size)
    prepared = copy.deepcopy(cache)
    trim_prompt_cache(prepared, prepared[0][0].offset - start)
    contexts = drafter.make_ctx_cache()
    if start:
        shape = (1, config.n_kv_heads, start, config.attn_head_dim)
        for context in contexts:
            # These rows are outside every draft query's trained window.
            # Retain their positions so the original mask and RoPE are exact.
            zeros = mx.zeros(shape, dtype=drafter.norm.weight.dtype)
            context.append(zeros, zeros)
    started = time.perf_counter()
    for position in range(start, end, step_size):
        piece = ids[position:min(position + step_size, end)]
        _, fused = target.prefill(mx.array([piece]), prepared, config.target_layer_ids,
                                  want_logits=False)
        drafter.update_context(fused, position, contexts)
        mx.eval([group.state for group in prepared], [context.k for context in contexts],
                [context.v for context in contexts])
        mx.clear_cache()
    equal = True
    for original, replayed in zip(cache, prepared):
        for old, new in zip(original.caches, replayed.caches):
            equal &= old.offset == new.offset
            if old.keys is not None:
                equal &= bool(mx.all(old.keys[..., :old.offset, :] == new.keys[..., :new.offset, :]))
                equal &= bool(mx.all(old.values[..., :old.offset, :] == new.values[..., :new.offset, :]))
    return prepared, contexts, {"replayed_tokens": end - start,
                                "seconds": time.perf_counter() - started,
                                "target_cache_bit_equal": equal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-source", type=Path, required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--drafter", type=Path, default=Path("artifacts/dspark-draft"))
    parser.add_argument("--prefix", type=Path, default=Path("artifacts/hermes-prefix.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/performance-hermes-dspark.json"))
    parser.add_argument("--caps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--modes", nargs="+", choices=["target", "mtp1", "mtp3"], default=["target", "mtp1", "mtp3"])
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--matrix-from-layer", type=int, help="Private comparison: use grouped experts for 5-16 rows at target sparse layers >= N")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.max_tokens < 512 or any(cap < 1 or cap > 8 for cap in args.caps):
        parser.error("Use at least 512 output tokens and draft caps from 1 to 8")
    if args.modes[0] != "target" or args.prefill_step_size < 1:
        parser.error("Run target first and use a positive prompt block size")
    if args.matrix_from_layer is not None and args.matrix_from_layer < 0:
        parser.error("--matrix-from-layer must be nonnegative")
    sys.path.insert(0, str(args.engine_source / "src"))
    from mlx_dspark.generate import speculative_generate
    from mlx_dspark.load import load_drafter

    args.memory_limit_gib, args.expert_cache_gib = 416, 32
    args.resident, args.mtp, args.no_optimize = True, any(mode != "target" for mode in args.modes), False
    model, tokenizer, _ = load_runtime(args)
    backbone = model.target if args.mtp else model
    selected = matrix_layers(backbone, args.matrix_from_layer)
    prefix, ids = load_prefix(args.prefix, args.model, True, args.mtp)
    if args.mtp:
        model.rebuild_draft_cache(prefix[-1], step_size=args.prefill_step_size)
    results = {"method": "Sequential 512-token runs on one resident model. Saved Hermes prefix. No cold prefill measurement.",
               "source": json.loads((args.drafter / "source.json").read_text()),
               "context_tokens": len(ids), "runs": []}
    if selected:
        results["private_matrix_comparison"] = {
            "first_target_layer": args.matrix_from_layer, "input_rows": [5, 16],
            "selected_target_layers": [index for index, _ in selected.values()],
            "requirements": "K4, resident, optimized",
            "same_prepared_cache_and_contexts": True,
            "target_wrappers_separate": True, "compiled_expert_calls_separate": True,
            "target_body_compiled": False,
            "grouped_source_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "exlr8/metal/fast_prefill.metal").read_bytes()).hexdigest(),
            "numeric_limit": "Matrix accumulation can change BF16 rounding versus native GEMV. Token equality is measured; it is not assumed.",
        }

    def save(row):
        results["runs"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items()
                          if key not in ("token_ids", "output", "cache_offsets", "accept_lengths",
                                         "affected_layer_shapes", "starting_target_cache_offsets",
                                         "starting_draft_context_shapes")}), flush=True)

    reference = None
    for mode in args.modes:
        cache = copy.deepcopy(prefix[:len(backbone.layers)] if mode == "target" else prefix)
        row, _ = benchmark(model, tokenizer, ids[-1:], mode, args.max_tokens, args.prefill_step_size, cache)
        row["prefill_reused"] = True
        row["completion_tokens_at_least_512"] = row["completion_tokens"] >= 512
        if mode == "target":
            reference = row["token_ids"]
        else:
            row["greedy_tokens_equal_target"] = row["token_ids"] == reference
            row["first_output_difference"] = first_difference(reference, row["token_ids"])
        save(row)

    drafter, config = load_drafter(str(args.drafter), quantize=False)
    target = wrap_target(backbone, tokenizer)
    drafter.bind_embed(target.draft_embed)
    drafter.bind_lm_head(target.lm_head_proj)
    cache, contexts, preparation = prepare_draft_prefix(target, drafter, prefix[:len(backbone.layers)], ids, args.prefill_step_size)
    results["draft_prefix_preparation"] = preparation
    print(json.dumps({"draft_prefix_preparation": preparation}), flush=True)
    if not preparation["target_cache_bit_equal"]:
        raise ValueError("Replayed target prefix differs; do not compare decode speeds")
    variants = [("native_expert_dispatch", target, None)]
    if selected:
        variants = [("native_gemv_compiled", target, compiled_expert_calls(selected, False)),
                    ("private_grouped_experts", wrap_target(backbone, tokenizer),
                     compiled_expert_calls(selected, True))]
    for cap in args.caps:
        baseline_ids = None
        for choice, run_target, calls in variants:
            run_cache, run_contexts = copy.deepcopy(cache), copy.deepcopy(contexts)
            initial_offsets = [group[0].offset for group in run_cache]
            initial_context_shapes = [list(context.k.shape) for context in run_contexts]
            active = mx.get_active_memory()
            mx.reset_peak_memory()
            patch = private_expert_calls(selected, calls) if calls is not None else nullcontext(Counter())
            started = time.perf_counter()
            with patch as evidence:
                generated = speculative_generate(
                    run_target, tokenizer, drafter, prompt_ids=ids, cache=run_cache,
                    ctx_caches=run_contexts, reuse_len=len(ids) - 1,
                    max_new_tokens=args.max_tokens, max_draft_tokens=cap, lookup_drafts=False,
                    temperature=0, apply_chat_template=False)
                mx.eval([group.state for group in run_cache], [context.k for context in run_contexts],
                        [context.v for context in run_contexts])
            elapsed = time.perf_counter() - started
            output_ids = generated.token_ids
            if baseline_ids is None:
                baseline_ids = list(output_ids)
            affected = [{"layer": index, "input_shape": list(x_shape), "expert_index_shape": list(id_shape),
                         "dtype": dtype, "calls": count}
                        for (index, x_shape, id_shape, dtype), count in sorted(evidence.items())]
            save({"mode": f"dspark{cap}" + (f"-matrix-from{args.matrix_from_layer}" if choice == "private_grouped_experts" else ""),
                  "context_tokens": len(ids), "kernel_choice": choice,
                  "matrix_from_layer": args.matrix_from_layer if choice == "private_grouped_experts" else None,
                  "compiled_expert_calls_separate": bool(selected),
                  "affected_layer_shapes": affected, "affected_layer_count": len({key[0] for key in evidence}),
                  "affected_expert_calls": sum(evidence.values()),
                  "starting_target_cache_offsets": initial_offsets,
                  "starting_draft_context_shapes": initial_context_shapes,
                  "completion_tokens": len(output_ids), "completion_tokens_at_least_512": len(output_ids) >= 512,
                  "prefill_reused": True, "prefill_step_size": args.prefill_step_size,
                  "prefill_seconds": generated.prefill_seconds, "elapsed_seconds": elapsed,
                  "decode_seconds": elapsed - generated.prefill_seconds,
                  "decode_tps": (len(output_ids) - 1) / (elapsed - generated.prefill_seconds),
                  "rounds": generated.num_rounds, "accept_lengths": generated.accept_lengths,
                  "mean_accept_length": sum(generated.accept_lengths) / max(generated.num_rounds, 1),
                  "greedy_tokens_equal_target": output_ids == reference,
                  "first_output_difference": first_difference(reference, output_ids),
                  "target_reference_completion_tokens": len(reference),
                  "greedy_tokens_equal_dspark_baseline": output_ids == baseline_ids,
                  "first_output_difference_vs_dspark_baseline": first_difference(baseline_ids, output_ids),
                  "dspark_baseline_completion_tokens": len(baseline_ids),
                  "token_ids": output_ids, "output": generated.text, "finish_reason": generated.finish_reason,
                  "active_gib": mx.get_active_memory() / 1024**3,
                  "peak_gib": mx.get_peak_memory() / 1024**3,
                  "peak_extra_gib": (mx.get_peak_memory() - active) / 1024**3,
                  "includes_first_compiled_call": bool(selected)})
            del run_cache, run_contexts
            mx.clear_cache()


if __name__ == "__main__":
    main()
