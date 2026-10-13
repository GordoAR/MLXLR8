"""Private exact-FP8 dense test. CPU checks do not import MLX.

The GPU test loads two dense projections from one layer and saved projection
inputs. It writes metrics only. It does not load or print prompt text.
"""

import argparse
from functools import lru_cache
import gc
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exlr8.static_cache import _e4m3_codes
from exlr8.trellis import mcg_codebook
from exlr8.weights import SafeTensorFile


ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = (1, 2, 4)


def records(args):
    file = SafeTensorFile(args.model / "dense" / f"exlr8-dense-{args.layer:03d}.safetensors")
    result = {}
    for name in ("q_b", "o"):
        record = tuple(map(int, file.raw(f"self_attn.{name}_proj.record")))
        if record[0] != 1:
            raise ValueError("this test requires a dense trellis projection")
        result[name] = record
    return file, result


def cpu_checks(args):
    header = (ROOT / "exlr8/metal/fast_decode_fp8.metal").read_text()
    bits = np.array([int(v, 16) for v in re.findall(r"0x([0-9a-f]{4})u", header)], np.uint16)
    assert bits.size == 256
    codes = np.arange(256, dtype=np.uint16)
    exponent, fraction = (codes >> 3) & 15, codes & 7
    magnitude = np.where(exponent == 0, fraction / 512,
                         np.ldexp(1 + fraction / 8, exponent.astype(np.int32) - 7))
    magnitude[(codes & 127) == 127] = np.nan
    expected = np.copysign(magnitude, np.where(codes & 128, -1, 1)).astype(np.float16)
    np.testing.assert_array_equal(bits, expected.view(np.uint16))
    book = mcg_codebook()
    np.testing.assert_array_equal(bits[_e4m3_codes(book)], book.view(np.uint16))

    file, source = records(args)
    inputs = SafeTensorFile(args.inputs)
    prefix = f"q4.native.layer{args.layer}."
    assert inputs.tensors[prefix + "q_norm"]["shape"] == [1, 4, source["q_b"][2]]
    value_shape = inputs.tensors[prefix + "value_expand"]["shape"]
    assert value_shape[0] == 1 and value_shape[2] == 4
    assert value_shape[1] * value_shape[3] == source["o"][2]
    geometry = []
    for name, record in source.items():
        _, out_features, in_features, *_ = record
        assert in_features % 256 == 0 and out_features % 128 == 0
        columns = np.array([base + lane * 8 + p * 4 + byte
                            for base in range(0, in_features, 256)
                            for lane in range(32) for p in range(2) for byte in range(4)])
        np.testing.assert_array_equal(np.sort(columns), np.arange(in_features))
        for outputs_per_simd in LAYOUTS:
            outputs_per_group = 2 * outputs_per_simd
            outputs = np.array([group * outputs_per_group + simd * outputs_per_simd + n
                                for group in range(out_features // outputs_per_group)
                                for simd in range(2) for n in range(outputs_per_simd)])
            np.testing.assert_array_equal(outputs, np.arange(out_features))
            for rows in (1, 2, 4):
                geometry.append({"projection": name, "queries": rows,
                                 "outputs_per_simd": outputs_per_simd,
                                 "threadgroup_threads": 64,
                                 "threadgroups": out_features // outputs_per_group,
                                 "each_weight_byte_read_once_per_call": True,
                                 "each_output_written_once": True})
    return {"all_256_fp8_half_patterns_equal": True,
            "all_65536_codebook_values_equal": True,
            "finite_fp8_codes": 254, "nan_codes": [127, 255],
            "signed_zero_patterns_equal": True,
            "input_shapes_match": True,
            "matrices": {name: {"in_features": r[2], "out_features": r[1],
                                 "expanded_bytes": r[1] * r[2],
                                 "rotating_weight_bytes": args.rotations * r[1] * r[2],
                                 "source_k": r[3], "sharded_side": r[4]}
                         for name, r in source.items()},
            "geometry": geometry, "mlx_imported": "mlx.core" in sys.modules}


@lru_cache(maxsize=24)
def kernel(in_features, out_features, rows, outputs_per_simd):
    import mlx.core as mx

    source = (ROOT / "tools/dense_unit_fp8.metal").read_text()
    header = (ROOT / "exlr8/metal/fast_decode_fp8.metal").read_text()
    prefix = (f"#define IN_FEATURES {in_features}u\n#define OUT_FEATURES {out_features}u\n"
              f"#define QUERY_ROWS {rows}u\n#define OUTPUTS_PER_SIMD {outputs_per_simd}u\n")
    return mx.fast.metal_kernel(
        name=f"exlr8_private_dense_unit_fp8_i{in_features}_o{out_features}_q{rows}_r{outputs_per_simd}",
        input_names=["xh", "packed"], output_names=["out"],
        header=header, source=prefix + source,
    )


def unit_inner(xh, packed, outputs_per_simd):
    import mlx.core as mx

    in_features = xh.shape[-1]
    rows = xh.size // in_features
    out_features = packed.shape[0]
    return kernel(in_features, out_features, rows, outputs_per_simd)(
        inputs=[xh.reshape(rows, in_features), packed],
        grid=(out_features // (2 * outputs_per_simd) * 64, 1, 1),
        threadgroup=(64, 1, 1), output_shapes=[(rows, out_features)],
        output_dtypes=[mx.float32],
    )[0].reshape(*xh.shape[:-1], out_features)


def unit_linear(x, packed, suh, svh, outputs_per_simd):
    import mlx.core as mx
    from exlr8.trellis import _hadamard

    xh = _hadamard(x.astype(mx.float32) * suh.astype(mx.float32))
    inner = unit_inner(xh, packed, outputs_per_simd)
    return (_hadamard(inner) * svh.astype(mx.float32)).astype(x.dtype)


def difference(actual, expected):
    import mlx.core as mx

    a, b = (np.asarray(v.astype(mx.float32)) for v in (actual, expected))
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {"finite": finite,
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b.astype(np.float64)), 1e-30)),
            "max_abs": float(np.max(np.abs(delta))),
            "changed_values": int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
            "values": int(a.size),
            "bit_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))}


def chain(call, x, weights, steps, feedback_scale):
    import mlx.core as mx

    state = x
    for i in range(steps):
        output = call(state, weights[i % len(weights)])
        feedback = mx.mean(output.astype(mx.float32), axis=-1, keepdims=True) * feedback_scale
        state = (state.astype(mx.float32) + feedback).astype(x.dtype)
    return output, state


def measure(call, x, weights, args):
    import mlx.core as mx

    for i in range(2):
        warm = call(x, weights[i % len(weights)])
        mx.eval(warm)
    del warm
    gc.collect()
    mx.clear_cache()
    active = mx.get_active_memory()
    mx.reset_peak_memory()
    value = call(x, weights[0])
    mx.eval(value)
    single_peak = mx.get_peak_memory() - active
    output_bytes = value.nbytes
    del value
    gc.collect()
    mx.clear_cache()
    samples, peaks, growth = [], [], []
    for _ in range(args.repeats):
        active = mx.get_active_memory()
        mx.reset_peak_memory()
        begin = time.perf_counter()
        value, state = chain(call, x, weights, args.steps, 1e-20)
        mx.eval(value, state)
        samples.append((time.perf_counter() - begin) * 1000 / args.steps)
        peaks.append(mx.get_peak_memory() - active)
        growth.append(mx.get_active_memory() - active)
        del value, state
        gc.collect()
        mx.clear_cache()
    return {"median_ms": float(np.median(samples)), "min_ms": min(samples),
            "samples_ms": samples, "chain_calls": args.steps,
            "rotating_matrices": len(weights), "timing_feedback_scale": 1e-20,
            "single_peak_extra_bytes": single_peak,
            "chain_peak_extra_bytes": max(peaks), "chain_active_extra_bytes": max(growth),
            "output_bytes": output_bytes}


def alternate(native, candidate, x, weights, args):
    one = argparse.Namespace(**vars(args))
    one.repeats = 1
    samples = {"native": [], "candidate": []}
    paired_speedups = []
    calls = {"native": native, "candidate": candidate}
    for repeat in range(args.repeats):
        order = ("native", "candidate") if repeat % 2 == 0 else ("candidate", "native")
        pair = {}
        for label in order:
            pair[label] = measure(calls[label], x, weights, one)["median_ms"]
            samples[label].append(pair[label])
        paired_speedups.append(pair["native"] / pair["candidate"])
    return {"order": "native,candidate; candidate,native; repeat",
            "pairs": args.repeats, "calls_per_sample": args.steps,
            "rotating_matrices": len(weights), "samples_ms": samples,
            "native_median_ms": float(np.median(samples["native"])),
            "candidate_median_ms": float(np.median(samples["candidate"])),
            "median_paired_speedup": float(np.median(paired_speedups)),
            "paired_speedups": paired_speedups}


def run(args, proof):
    import mlx.core as mx
    from exlr8.dense_trellis import expand_trellis, expanded_linear
    from exlr8.trellis import _hadamard

    mx.set_wired_limit(16 * 1024**3)
    file, source = records(args)
    saved = SafeTensorFile(args.inputs)
    result = {"layer": args.layer, "input_source": args.inputs.name,
              "cpu_checks": proof, "dtype_cases": ["float32", "bfloat16"],
              "rows": [1, 2, 4], "chain_feedback_scale": 0.001,
              "weights": "exact expanded uint32 E4M3 bytes; exponent 127",
              "outer_transforms": "native F32 scales and native Hadamard", "cases": []}
    screen = json.loads(args.confirm.read_text()) if args.confirm else None
    if screen is not None:
        result["screen_source"] = args.confirm.name
    for name, record in source.items():
        _, out_features, in_features, k, side, *_ = record
        prefix = f"self_attn.{name}_proj."
        suh, svh = file.load(prefix + "suh"), file.load(prefix + "svh")
        packed, exponents = expand_trellis(file.load(prefix + "trellis"), k, in_features, out_features, side)
        mx.eval(packed, exponents, suh, svh)
        assert bool(mx.all(exponents == 127))
        owner = mx.contiguous(mx.broadcast_to(packed[None], (args.rotations, *packed.shape)))
        weights = tuple(owner[i] for i in range(args.rotations))
        mx.eval(weights)
        del packed, owner
        input_key = f"q4.native.layer{args.layer}." + ("q_norm" if name == "q_b" else "value_expand")
        original = saved.load(input_key)
        if name == "o":
            original = original.swapaxes(1, 2).reshape(1, 4, in_features)
        mx.eval(original)
        native = mx.compile(lambda x, w: expanded_linear(x, w, exponents, suh, svh))
        custom = {f"unit_r{r}": mx.compile(lambda x, w, r=r: unit_linear(x, w, suh, svh, r)) for r in LAYOUTS}
        for rows in (1, 2, 4):
            for dtype_name in ("float32", "bfloat16"):
                x = original[:, :rows].astype(getattr(mx, dtype_name))
                mx.eval(x)
                if screen is not None:
                    case = next(c for c in screen["cases"] if c["projection"] == name
                                and c["queries"] == rows and c["input_dtype"] == dtype_name)
                    label = min(custom, key=lambda n: case["variants"][n]["median_ms"])
                    entry = {"projection": name, "queries": rows, "input_dtype": dtype_name,
                             "candidate": label, "selection": "lowest screen median",
                             **alternate(native, custom[label], x, weights, args)}
                    result["cases"].append(entry)
                    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                    print(json.dumps(entry, allow_nan=False), flush=True)
                    del x
                    continue
                reference = native(x, weights[0])
                xh = _hadamard(x.astype(mx.float32) * suh.astype(mx.float32))
                inner_reference = mx.quantized_matmul(xh, weights[0], scales=exponents, mode="mxfp8")
                ref_chain, ref_state = chain(native, x, weights, args.steps, 0.001)
                mx.eval(reference, inner_reference, ref_chain, ref_state)
                entry = {"projection": name, "queries": rows, "input_dtype": dtype_name,
                         "expanded_bytes": in_features * out_features, "variants": {}}
                for label, call in {"native": native, **custom}.items():
                    actual = call(x, weights[0])
                    serial = call(x[:, :1], weights[0])
                    final, state = chain(call, x, weights, args.steps, 0.001)
                    mx.eval(actual, serial, final, state)
                    metrics = {"output_error": difference(actual, reference),
                               "first_row_vs_serial": difference(actual[:, :1], serial),
                               "chain75_output_error": difference(final, ref_chain),
                               "chain75_input_state_error": difference(state, ref_state)}
                    if label != "native":
                        inner = unit_inner(xh, weights[0], int(label[-1]))
                        mx.eval(inner)
                        metrics["inner_f32_error"] = difference(inner, inner_reference)
                        del inner
                    del actual, serial, final, state
                    metrics.update(measure(call, x, weights, args))
                    entry["variants"][label] = metrics
                baseline_ms = entry["variants"]["native"]["median_ms"]
                for label, metrics in entry["variants"].items():
                    metrics["speedup_vs_native"] = baseline_ms / metrics["median_ms"]
                result["cases"].append(entry)
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                print(json.dumps(entry, allow_nan=False), flush=True)
                del reference, xh, inner_reference, ref_chain, ref_state, x
        del native, custom, original, weights, exponents, suh, svh
        gc.collect()
        mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / ".lmstudio/models/kindlingai/glm-5.3-exlr8-up1")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/layer1-operators.safetensors"))
    parser.add_argument("--rotations", type=int, default=21)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--confirm", type=Path, help="repeat each screen winner with alternating native/candidate calls")
    parser.add_argument("--output", type=Path, default=Path("tools/dense_unit_fp8_results.json"))
    args = parser.parse_args()
    if args.rotations < 2 or args.steps < 1 or args.repeats < 1:
        parser.error("need rotations >= 2 and positive steps/repeats")
    proof = cpu_checks(args)
    if args.cpu_check:
        print(json.dumps(proof, indent=2), flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args, proof)


if __name__ == "__main__":
    main()
