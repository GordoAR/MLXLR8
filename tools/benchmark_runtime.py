"""Measure one resident model at a time, then keep that model as the API."""

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import stream_generate as target_generate
from mlx_lm.models.cache import CacheList, LRUPromptCache, load_prompt_cache, save_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from exlr8.cli import DEFAULT_MODEL, load_runtime, serve
from exlr8.mtp import MTPState, stream_generate as mtp_generate


def save_prefix(path, cache, ids, model_path, optimized):
    """Save the prepared prefix outside the timed decode run."""
    serializable = [CacheList(*group.caches) if isinstance(group, MTPState) else group
                    for group in cache]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_prompt_cache(str(path), serializable, metadata={
        "model": str(Path(model_path).resolve()), "prompt_ids": json.dumps(ids),
        "optimized": str(optimized), "mtp": str(isinstance(cache[-1], MTPState)),
    })


def load_prefix(path, model_path, optimized, mtp):
    cache, metadata = load_prompt_cache(str(path), return_metadata=True)
    if metadata["model"] != str(Path(model_path).resolve()) or metadata["optimized"] != str(optimized):
        raise ValueError("The saved prefix uses a different model or kernel mode")
    ids = json.loads(metadata["prompt_ids"])
    saved_mtp = metadata["mtp"] == "True"
    if mtp and not saved_mtp:
        raise ValueError("MTP tests need a saved prefix with the MTP state")
    if saved_mtp:
        cache[-1] = MTPState(*cache[-1].caches)
        if not mtp:
            cache = cache[:-1]
    if any(group[0].offset != len(ids) - 1 for group in cache):
        raise ValueError("The saved prefix offset does not match its prompt")
    mx.eval([group.state for group in cache])
    return cache, ids


def benchmark(model, tokenizer, ids, mode, limit, step_size, cache=None):
    target = model.target if hasattr(model, "target") else model
    runtime = target if mode == "target" else model
    generate = target_generate if mode == "target" else mtp_generate
    kwargs = {} if mode == "target" else {"num_draft_tokens": int(mode[3:])}
    cache = runtime.make_cache() if cache is None else cache
    cached_tokens = cache[0][0].offset
    progress = []
    started = time.perf_counter()
    events = []
    text = []
    def update_progress(done, total):
        progress.append((done, total, time.perf_counter() - started))
        if total >= 4096 and (done == total or done % 1024 == 0):
            print(json.dumps({"mode": mode, "prefill_processed": done, "prefill_total": total,
                              "seconds": time.perf_counter() - started}), flush=True)

    for response in generate(
        runtime, tokenizer, ids, max_tokens=limit, sampler=make_sampler(0),
        prefill_step_size=step_size, prompt_cache=cache,
        prompt_progress_callback=update_progress,
        **kwargs,
    ):
        events.append((response.token, time.perf_counter() - started))
        text.append(response.text)
    elapsed = time.perf_counter() - started
    first_ready = progress[-1][2]
    result = {
        "mode": mode, "prompt_tokens": len(ids), "cached_tokens": cached_tokens,
        "prefill_step_size": step_size,
        "context_tokens": cached_tokens + len(ids),
        "completion_tokens": len(events), "finish_reason": response.finish_reason,
        "prefill_seconds": first_ready, "prefill_tps": len(ids) / first_ready,
        "ttft_seconds": events[0][1], "elapsed_seconds": elapsed,
        "decode_seconds": elapsed - first_ready,
        "decode_tps": (len(events) - 1) / max(elapsed - first_ready, 1e-9),
        "token_ids": [event[0] for event in events], "output": "".join(text),
        "active_gib": mx.get_active_memory() / 1024**3,
        "peak_gib": mx.get_peak_memory() / 1024**3,
        "cache_offsets": [group[0].offset for group in cache],
    }
    if mode != "target":
        result.update(model._mtp_stats)
    mx.clear_cache()
    return result, cache


def profile_layers(target, cache, token):
    cache = cache[:len(target.layers)]
    durations = []
    previous = time.perf_counter()

    def trace(index, hidden):
        nonlocal previous
        mx.eval(hidden)
        current = time.perf_counter()
        durations.append({"layer": index, "milliseconds": (current - previous) * 1000})
        previous = current

    target.trace = trace
    try:
        logits = target(mx.array([[token]]), cache=cache)
        before_head_eval = time.perf_counter()
        mx.eval(logits)
        head_ms = (time.perf_counter() - before_head_eval) * 1000
    finally:
        target.trace = None
    return {"method": "One decode token; synchronized after each layer. Diagnostic only.",
            "layers": durations, "head_milliseconds": head_ms}


def compare_verification(target, cache, tokens, prompt_tokens, difference, layer_trace=False):
    """Compare four-row and one-row target calls on the same real prefix."""
    start = max(difference - 1, 0)
    inputs = tokens[start:start + 4]
    batched_cache, serial_cache = copy.deepcopy(cache), copy.deepcopy(cache)
    for item in (batched_cache, serial_cache):
        trim_prompt_cache(item, item[0][0].offset - prompt_tokens - start)
    captures = {"batch": {}, "serial": {}}
    if layer_trace:
        from exlr8 import model as model_module
        original_layer, original_router = model_module.DecoderLayer.__call__, model_module.route_experts
        layer_numbers = {id(layer): index for index, layer in enumerate(target.layers)}
        current_layer = None
        routes = {}

        def capture_router(*args):
            result = original_router(*args)
            routes[current_layer] = result[0]
            return result

        def capture_layer(layer, *args, **kwargs):
            nonlocal current_layer
            current_layer = layer_numbers[id(layer)]
            result = original_layer(layer, *args, **kwargs)
            captures[branch].setdefault(current_layer, []).append((*result, routes.get(current_layer)))
            return result

        model_module.DecoderLayer.__call__, model_module.route_experts = capture_layer, capture_router
    try:
        branch = "batch"
        batched = target(mx.array([inputs], dtype=mx.uint32), cache=batched_cache)[0]
        mx.eval(batched)
        serial = []
        branch = "serial"
        for token in inputs:
            logits = target(mx.array([[token]], dtype=mx.uint32), cache=serial_cache)[0, 0]
            mx.eval(logits)
            serial.append(logits)
    finally:
        if layer_trace:
            model_module.DecoderLayer.__call__, model_module.route_experts = original_layer, original_router
    batched, serial = batched.astype(mx.float32), mx.stack(serial).astype(mx.float32)
    batch_logp = batched - mx.logsumexp(batched, axis=-1, keepdims=True)
    serial_logp = serial - mx.logsumexp(serial, axis=-1, keepdims=True)
    result = {
        "prefix_tokens": prompt_tokens + start, "verification_rows": len(inputs),
        "first_output_difference": difference,
        "batch_argmax": mx.argmax(batched, axis=-1).tolist(),
        "serial_argmax": mx.argmax(serial, axis=-1).tolist(),
        "saved_target_next": tokens[start + 1:start + 1 + len(inputs)],
        "relative_l2": float(mx.linalg.norm(batched - serial) / mx.linalg.norm(serial)),
        "max_logit_error": float(mx.max(mx.abs(batched - serial))),
        "mean_kl_serial_to_batch": float(mx.mean(mx.sum(
            mx.exp(serial_logp) * (serial_logp - batch_logp), axis=-1))),
    }
    if layer_trace:
        causal_cache = copy.deepcopy(cache)
        trim_prompt_cache(causal_cache, causal_cache[0][0].offset - prompt_tokens - start)
        changed_inputs = [inputs[0], *[int(token) ^ 1 for token in inputs[1:]]]
        changed = target(mx.array([changed_inputs], dtype=mx.uint32), cache=causal_cache)[0, 0]
        mx.eval(changed)
        result["first_token_unchanged_when_future_inputs_change"] = bool(mx.all(changed == batched[0]))
        result["cache_offsets_match"] = (
            [item.offset for group in batched_cache for item in group.caches]
            == [item.offset for group in serial_cache for item in group.caches])
        result["layers"] = []
        for index in range(len(target.layers)):
            batch_hidden, batch_indices, batch_ids = captures["batch"][index][0]
            serial_hidden = mx.concatenate([row[0] for row in captures["serial"][index]], axis=1)
            a, b = batch_hidden[0, 0].astype(mx.float32), serial_hidden[0, 0].astype(mx.float32)
            row = {"layer": index, "first_token_relative_l2": float(mx.linalg.norm(a - b) / mx.linalg.norm(b)),
                   "first_token_max_abs": float(mx.max(mx.abs(a - b)))}
            if batch_indices is not None:
                serial_indices = captures["serial"][index][0][1]
                row["first_token_dsa_set_equal"] = bool(mx.all(
                    mx.sort(batch_indices[:, :, 0], axis=-1) == mx.sort(serial_indices[:, :, 0], axis=-1)))
            if batch_ids is not None:
                serial_ids = captures["serial"][index][0][2]
                row["first_token_expert_set_equal"] = bool(mx.all(
                    mx.sort(batch_ids[:, 0], axis=-1) == mx.sort(serial_ids[:, 0], axis=-1)))
            result["layers"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--memory-limit-gib", type=float, default=416)
    parser.add_argument("--expert-cache-gib", type=float, default=32)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--splitk-verify", action="store_true", help="Test the private two-to-four-row attention path")
    parser.add_argument("--serial-dense-mlp", action="store_true", help="Test one-row dense MLP products during verification")
    parser.add_argument("--unit-dense", action="store_true", help="Test unit-exponent FP8 query and output projections")
    prefill = parser.add_mutually_exclusive_group()
    prefill.add_argument("--indexed-prefill", action="store_true", default=True, help="Use indexed attention for long sparse prompt blocks (default)")
    prefill.add_argument("--native-prefill", action="store_false", dest="indexed_prefill", help="Use native sparse attention for the comparison run")
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prefill-step-sizes", nargs="+", type=int, help="Compare prompt block sizes on one loaded model")
    parser.add_argument("--modes", nargs="+", default=["target", "mtp1", "mtp3"])
    parser.add_argument("--input-lengths", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-file", type=Path, help="Hermes request JSON with messages and tools, or system prompt text")
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--followup-file", type=Path, help="Add a source file to the cached follow-up request")
    parser.add_argument("--reuse-prefill", action="store_true", help="Compare decode modes from copies of one prepared prompt cache")
    parser.add_argument("--save-prefix", type=Path, help="Save the prepared prefix for later decode diagnostics")
    parser.add_argument("--load-prefix", type=Path, help="Use a saved prefix; skip cold prefill")
    parser.add_argument("--output", type=Path, default=Path("artifacts/performance-optimized.json"))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--layer-fidelity", action="store_true", help="Record layer differences between batch and serial verification")
    parser.add_argument("--trace-gpu", action="store_true", help="Wait for an external profiler, then run a separate target decode")
    parser.add_argument("--route-trace", type=Path, help="Export target decode routes and FP8-cache hit estimates")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    step_sizes = args.prefill_step_sizes or [args.prefill_step_size]
    if any(mode != "target" and (not mode.startswith("mtp") or not mode[3:].isdigit()) for mode in args.modes):
        parser.error("Modes must be target or mtp followed by a draft token count")
    if args.max_tokens < 1 or min(step_sizes) < 1:
        parser.error("Token counts must be positive")
    if (args.save_prefix or args.load_prefix) and len(args.input_lengths) != 1:
        parser.error("Saved prefix tests require one input length")
    if args.save_prefix and len(step_sizes) != 1:
        parser.error("Saving a prefix requires one prompt block size")
    if args.trace_gpu and not (args.reuse_prefill or args.save_prefix or args.load_prefix):
        parser.error("GPU tracing requires a prepared prefix")
    args.resident = True
    args.mtp = any(mode != "target" for mode in args.modes)
    args.num_draft_tokens = 3
    args.host = "127.0.0.1"
    args.temp = 0
    runtime = load_runtime(args)
    model, tokenizer, store = runtime
    target = model.target if args.mtp else model
    if args.unit_dense:
        from exlr8.dense import DenseLinear
        from tools.bench_dense_unit_fp8 import unit_linear

        projections = {id(getattr(layer.self_attn, name)): name for layer in target.layers
                       for name in ("q_b_proj", "o_proj")
                       if getattr(layer.self_attn, name).format == 1}
        original_unit_linear = DenseLinear.__call__

        def unit_dense(linear, x):
            name = projections.get(id(linear))
            rows = x.size // linear.in_features
            if name is not None and 1 <= rows <= 4:
                outputs_per_simd = 4 if name == "q_b_proj" and rows <= 2 else 2
                return unit_linear(x, linear.fp8_packed, linear.suh, linear.svh, outputs_per_simd)
            return original_unit_linear(linear, x)

        DenseLinear.__call__ = unit_dense
    if args.serial_dense_mlp:
        from exlr8.dense import DenseLinear

        dense_ids = {id(getattr(layer, name)) for layer in target.layers if layer.experts is None
                     for name in ("gate_proj", "up_proj", "down_proj")}
        original_linear = DenseLinear.__call__

        def serial_dense_mlp(linear, x):
            if id(linear) in dense_ids and 2 <= x.shape[-2] <= 4:
                return mx.concatenate([original_linear(linear, x[..., i:i + 1, :])
                                       for i in range(x.shape[-2])], axis=-2)
            return original_linear(linear, x)

        DenseLinear.__call__ = serial_dense_mlp
    if args.splitk_verify:
        from exlr8.dsa_prefill_splitk import install_splitk_control

        for layer in target.layers:
            install_splitk_control(layer.self_attn)
    padding = tokenizer.encode("These notes provide neutral context for a speed test. ", add_special_tokens=False)
    instruction = "Print the consecutive integers from 1 to 300, one per line. Start now."
    tools = None
    system = None
    if args.prompt_file is not None:
        source = args.prompt_file.read_text()
        if args.prompt_file.suffix == ".json":
            request = json.loads(source)
            if "messages" not in request and "request" in request:
                request = request["request"]
            request = request.get("body", request)
            system = request.get("instructions") or "\n\n".join(
                message["content"] for message in request.get("messages", [])
                if message["role"] == "system" and isinstance(message["content"], str)
            )
            tools = request.get("tools")
            if tools and "function" not in tools[0]:
                tools = [{"type": "function", "function": {key: tool[key] for key in
                         ("name", "description", "parameters", "strict") if key in tool}}
                         for tool in tools if tool.get("type") == "function"]
        else:
            system = source
        instruction = (
            "Review the following inference code. Explain the performance risks, accuracy checks, "
            "and a complete test plan. Use the supplied code. Write at least 800 words. "
            "No tool use is needed for this review.\n\n"
            + (Path(__file__).resolve().parents[1] / "exlr8/experts.py").read_text()
        )
    base_messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": instruction}]
    tail = tokenizer.apply_chat_template(
        base_messages, tools=tools, tokenize=True, add_generation_prompt=True, reasoning_effort="low",
    )
    result = {
        "method": "Sequential native runs. One full resident model. Cache reuse is marked per run. Greedy sampler. Decode excludes the first token and includes cache finalization.",
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "prefill_step_size": step_sizes[0] if len(step_sizes) == 1 else None,
        "prefill_step_sizes": step_sizes, "dense_optimized": not args.no_optimize,
        "splitk_verification": args.splitk_verify,
        "direct_activation_loads": True,
        "serial_dense_mlp": args.serial_dense_mlp,
        "unit_dense": args.unit_dense,
        "indexed_prefill": args.indexed_prefill and not args.no_optimize,
        "resident_weights": store.stats(), "runs": [],
        "prompt_source": str(args.prompt_file) if args.prompt_file else "Synthetic counting prompt",
        "followup_source": str(args.followup_file) if args.followup_file else None,
        "system_tokens": len(tokenizer.encode(system, add_special_tokens=False)) if system else 0,
        "tools": len(tools or []),
    }
    warmup_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 50, one number per line."}],
        tokenize=True, add_generation_prompt=True, reasoning_effort="low",
    )
    benchmark(model, tokenizer, warmup_ids, "target", 8, step_sizes[0])
    last_cache = None
    last_token = None
    for length, step_size in ((n, s) for n in args.input_lengths for s in step_sizes):
        args.prefill_step_size = step_size
        length = max(length, len(tail))
        fill = (padding * ((length + len(padding) - 1) // len(padding)))[:length - len(tail)]
        ids = [*fill, *tail]
        shared_cache = None
        if args.load_prefix:
            shared_cache, ids = load_prefix(args.load_prefix, args.model, not args.no_optimize, args.mtp)
            result["loaded_prefix"] = str(args.load_prefix)
            if args.mtp:
                model.rebuild_draft_cache(shared_cache[-1], args.prefill_step_size)
                result["mtp_draft_cache_rebuilt"] = True
        elif args.reuse_prefill or args.save_prefix:
            prefill, shared_cache = benchmark(
                model, tokenizer, ids, "mtp1" if args.mtp else "target", 1, args.prefill_step_size)
            prefill["mode"] = "shared_prefill"
            result.setdefault("prefill_runs", []).append(prefill)
            trim_prompt_cache(shared_cache, shared_cache[0][0].offset - len(ids) + 1)
            if args.save_prefix:
                save_prefix(args.save_prefix, shared_cache, ids, args.model, not args.no_optimize)
        previous = None
        reference_cache = None
        for mode in args.modes:
            trace = None
            if args.route_trace is not None and mode == "target":
                from exlr8.experts import ExpertLayer
                from exlr8.fast_cache import RouteTrace

                trace = RouteTrace(token_limit=args.max_tokens)
                original_call = ExpertLayer.__call__

                def traced_call(layer, x, selected):
                    trace.record(layer.number, selected)
                    return original_call(layer, x, selected)

                ExpertLayer.__call__ = traced_call
            try:
                if shared_cache is None:
                    run_ids, run_cache = ids, None
                else:
                    run_ids = ids[-1:]
                    run_cache = copy.deepcopy(shared_cache[:len(target.layers)] if mode == "target" else shared_cache)
                row, cache = benchmark(model, tokenizer, run_ids, mode, args.max_tokens,
                                       args.prefill_step_size, cache=run_cache)
            finally:
                if trace is not None:
                    ExpertLayer.__call__ = original_call
            if trace is not None:
                args.route_trace.parent.mkdir(parents=True, exist_ok=True)
                trace.write(args.route_trace)
            if shared_cache is not None:
                row["prefill_reused"] = True
                row["benchmark_prompt_tokens"] = len(ids)
            if mode == "target":
                previous = row["token_ids"]
                reference_cache = cache if args.profile or args.layer_fidelity else None
            elif previous is not None:
                row["greedy_tokens_equal_target"] = row["token_ids"] == previous
                row["first_output_difference"] = next(
                    (i for i, (a, b) in enumerate(zip(previous, row["token_ids"])) if a != b), None)
                if (args.profile or args.layer_fidelity) and not row["greedy_tokens_equal_target"]:
                    difference = row["first_output_difference"]
                    if difference is None:
                        difference = min(len(previous), len(row["token_ids"]))
                    result.setdefault("batch_fidelity", []).append(compare_verification(
                        target, reference_cache, previous, len(ids), difference, args.layer_fidelity))
            result["runs"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: value for key, value in row.items() if key not in ("token_ids", "output", "cache_offsets")}), flush=True)
            last_cache, last_token = cache, row["token_ids"][-1]
            if args.followup:
                followup_content = "Continue the review. Give the detailed test cases and explain how to check each result."
                if args.followup_file is not None:
                    followup_content += "\n\nReview this additional code:\n\n" + args.followup_file.read_text()
                messages = [*base_messages, {"role": "assistant", "content": row["output"]},
                            {"role": "user", "content": followup_content}]
                next_ids = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                                         add_generation_prompt=True, reasoning_effort="low")
                lru = LRUPromptCache(max_size=1)
                lru.insert_cache(mode, [*ids, *row["token_ids"]], cache)
                restored, rest = lru.fetch_nearest_cache(mode, next_ids)
                followup, cache = benchmark(model, tokenizer, rest, mode, args.max_tokens,
                                            args.prefill_step_size, cache=restored)
                followup["followup"] = True
                result["runs"].append(followup)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({key: value for key, value in followup.items() if key not in ("token_ids", "output", "cache_offsets")}), flush=True)
                last_cache, last_token = cache, followup["token_ids"][-1]
    if args.profile:
        result["profile"] = profile_layers(target, last_cache, last_token)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["profile"]), flush=True)
    if args.trace_gpu:
        print(json.dumps({"trace_pid": os.getpid(), "profile_phase": "waiting", "diagnostic_only": True}), flush=True)
        input()
        diagnostic_cache = copy.deepcopy(shared_cache[:len(target.layers)])
        diagnostic, _ = benchmark(model, tokenizer, ids[-1:], "target", args.max_tokens,
                                  args.prefill_step_size, cache=diagnostic_cache)
        diagnostic["diagnostic_only"] = True
        result["gpu_trace_run"] = diagnostic
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.serve:
        serve(args, runtime=runtime)


if __name__ == "__main__":
    main()
