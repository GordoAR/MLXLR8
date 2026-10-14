# MLXLR8

Run GLM-5.3 EXLR8 **up1** weights on Apple Silicon with MLX. MLXLR8 adds the EXLR8 loader, MLX integration, and tested kernel changes. The Python package and terminal command are named `exlr8`.

The runtime serves a local OpenAI-compatible chat API. All 78 main layers run. MTP is off by default.

Optimized resident mode converts the 75 main expert layers to native MLX MXFP4 groups of 32 in RAM. Gate and up projections share input preparation and one joined matrix operation. Input scale magnitudes are included before quantization. Signed transforms, output scales, and activation casts are retained. This conversion changes weight values. Model files remain unchanged. SSD-cache mode keeps the original packed trellis weights. `--no-optimize` selects the original expert values and reference operators.

## Requirements

- Apple Silicon Mac with **macOS 26 or later**. The pinned MLX wheels require this version.
- Native **Python 3.11** and Git. Check that Python reports `arm64`.
- A complete local **GLM-5.3 EXLR8 up1** model folder from [KindlingAI's EXLR8 checkpoint](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4), with the files below.
- For the tested resident profile: **M3 Ultra with 512 GB unified memory**. Use a **440 GiB** memory limit. Leave memory for macOS and other apps.
- For SSD-cache mode: enough RAM for your chosen cache, dense weights, prompt buffers, and macOS. This mode reads selected experts from disk and is slower.

The tested system used macOS 27.0.1, Python 3.11, MLX 0.32.3, and MLX-LM 0.32.0. Other hardware profiles have not received the same full-model checks.

## Download and install

```sh
git clone https://github.com/GordoAR/MLXLR8.git
cd MLXLR8
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements-macos-arm64.txt .
python -m pip check
python -c 'import platform; assert platform.machine() == "arm64"; import mlx.core as mx; print(mx.device_info())'
exlr8 --help
```

You can also download the repository ZIP from GitHub, extract it, and run the install steps inside that folder. Use the supplied constraints file to retain the tested dependency versions. Run these commands in a native terminal, without Rosetta.

## Set the model path

Set `MODEL` to the folder that contains `config.json`. Quote the path if it contains spaces.

```sh
export MODEL="/absolute/path/to/glm-5.3-exlr8-up1"
```

Required layout for normal generation:

```text
MODEL/
  config.json
  tokenizer.json
  tokenizer_config.json
  manifests/exlr8-manifest-up1.json
  dense/exlr8-vocab.safetensors
  dense/exlr8-dense-000.safetensors ... exlr8-dense-077.safetensors
  experts/exlr8-layer-003.k4.safetensors ... exlr8-layer-077.k4.safetensors
  experts/exlr8-layer-003.index.safetensors ... exlr8-layer-077.index.safetensors
```

Keep the model's chat-template files and license with the download. The manifest must be named `up1` and select uniform K4 weights for all main expert layers. `--mtp` also needs the retained layer 78 dense, K3 expert, and index files.

Check the two main metadata files before loading:

```sh
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["MODEL"])
config = json.loads((p / "config.json").read_text())
manifest = json.loads((p / "manifests/exlr8-manifest-up1.json").read_text())
assert config["model_type"] == "glm_moe_dsa"
assert config["num_hidden_layers"] == 78 and config["hidden_size"] == 6144
assert manifest["name"] == "up1"
print("Model metadata: OK")
PY
```

## Generate a response

On the tested 512 GB Mac:

```sh
MLX_MAX_MB_PER_BUFFER=2048 exlr8 generate \
  --model "$MODEL" --resident --memory-limit-gib 440 \
  --prompt "What is 2 + 2? Give a short answer." --max-tokens 64 --temp 0
```

Loading the resident model takes time. Progress and memory use go to stderr; generated text goes to stdout. Run one resident model at a time.

An SSD-cache example uses a 32 GiB expert cache and a 96 GiB MLX memory limit:

```sh
MLX_MAX_MB_PER_BUFFER=2048 exlr8 generate \
  --model "$MODEL" --expert-cache-gib 32 --memory-limit-gib 96 \
  --prefill-step-size 512 \
  --prompt "What is 2 + 2? Give a short answer." --max-tokens 64 --temp 0
```

Adjust those limits to your Mac's RAM. The MLX limit is a process budget, not additional memory. Prompt and cache use increase with context length.

## Start the local API

```sh
MLX_MAX_MB_PER_BUFFER=2048 exlr8 serve \
  --model "$MODEL" --resident --memory-limit-gib 440 \
  --host 127.0.0.1 --port 8081
```

For SSD-cache mode, replace `--resident --memory-limit-gib 440` with `--expert-cache-gib 32 --memory-limit-gib 96 --prefill-step-size 512`.

Wait for the model to load. In a second terminal:

```sh
curl -sS http://127.0.0.1:8081/v1/models
curl -sS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-exlr8-up1","messages":[{"role":"user","content":"What is 2 + 2? Give a short answer."}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

Client settings: base URL `http://127.0.0.1:8081/v1`, model `glm-5.3-exlr8-up1`. `/v1/models` lists the local model path; the chat API also accepts this stable alias. Use `"stream":true` in a chat request for streamed output. Press Ctrl-C in the server terminal to stop it before starting another resident process.

## Runtime settings

EXLR8 selects `MLX_MAX_OPS_PER_BUFFER=100` before MLX loads. An explicit value takes precedence. The commands above select `MLX_MAX_MB_PER_BUFFER=2048`; MLX counts array elements for this setting despite its name.

Optimized resident mode defaults to a 440 GiB process budget and requires at least 416 GiB. The reference resident mode defaults to 416 GiB and requires at least 384 GiB. Long prompts and larger input blocks need more temporary memory. MXFP4 stores four-bit values in groups of 32 with a shared scale exponent. Conversion occurs during model loading and does not write to the checkpoint. The optional draft layer keeps its original K3 format.

Rebuild saved optimized benchmark prefixes from earlier versions. Their cached states use different expert values. The benchmark tool records the expert format and rejects old optimized prefixes. Saved reference prefixes remain compatible.

Resident mode retains dense K4 weights, including with `--mtp`. One to three BF16 token rows use the K4 path; four-row queries share reads from the existing FP8 weights. The 75 eligible four-row output projections use an additional MXFP4 representation. This adds rounding and about 3.73 GiB at the measured peak. Other row counts retain their existing operators. Residual-add/RMS fusion and attention lookup support one to four BF16 token rows in a batch of one. RMS fusion uses the verified M3 Ultra, 6,144-channel profile. Larger token batches use the existing operators. `--no-optimize` selects the reference operators for diagnosis. MTP retains the production settings and uses the production prefill input sizes. With `--mtp`, matching spans from prior prompt and generated tokens start with up to three proposals. Full acceptance increases the limit to 15, then to 63. Rejection restores the three-proposal limit. The full target model verifies every proposal. The K3 layer provides up to three proposals when no useful prompt match is available. `--num-draft-tokens` sets the maximum for either source. MTP measures the ordinary generation pipeline with up to eight tokens and compares each source's cost per committed token. It checks four K3 cost samples before it stops a costly neural source. The existing word acceptance rule is retained. It uses ordinary generation when proposals cost more and continues to search for useful prompt matches. One to four target rows use the shared dense, compiled expert finish, and indexed sparse attention paths. MTP remains off by default.

## Checks and measured results

Install the test extra and run the portable suite from the repository folder:

```sh
python -m pip install -c requirements-macos-arm64.txt '.[test]'
EXLR8_TEST_METAL=1 python -m pytest -q
```

The repository includes PonyExl3's small CPU reference files for these tests. Tests that need private captured tensors skip when those fixtures are absent. No model download is needed for the portable tests.

Version 0.1.0 passed 621 tests and 23 subtests with Metal tests enabled. Nine tests were skipped. The clean release package also passed normal and streamed chat requests with the local up1 model.

The combined production test used two reversed pairs. Each process generated 512 outputs from a saved 23,861-token prefix, then 512 outputs after the same 4,469 new input tokens.

| Request | Previous tokens/s | Production tokens/s | Less decode time |
|---|---:|---:|---:|
| First | 13.61 | 13.76 | 1.12% |
| Follow-up | 13.52 | 13.66 | 1.04% |

Both pairs were faster. All eight requests matched tokens, text, full log probabilities, and allocated cache values exactly. Total follow-up time fell by 1.17%. Follow-up peak memory increased by 2.56 GiB to 385.26 GiB. These figures describe this test; speed depends on hardware, prompt, and context. Exactness is relative to the previous EXLR8 runtime using the same quantized weights.

The MTP tuning test completed one normal-generation control and two fresh processes per MTP limit. Each reply used 512 output tokens. The final control was stopped at the user's request. All runs used the same cached prompt and follow-up input. Target row counts were compiled before timing.

| Mode | First reply tokens/s | Follow-up tokens/s | Total follow-up seconds |
|---|---:|---:|---:|
| Production | 13.78 | 13.67 | 78.22 |
| MTP, draft limit one | 13.50 | 15.63 | 74.12 |
| MTP, draft limit three | 12.05 | 18.10 | 69.64 |

Dense K4 batches now share weight reads. Small batches use the compiled fused expert result step and direct indexed sparse attention. MTP compares measured cost per committed token and stops drafting when it costs more.

Against the preceding batched suite, the one-token limit increased first-reply speed by 22.0% and follow-up speed by 19.6%. The three-token limit increased these speeds by 29.9% and 17.5%. Its combined first-reply and follow-up time fell by 13.5%. The old and new suites were separate; the completed normal control stayed close to the old speeds. Against that new control, combined request time fell by 2.9%. Normal generation remains faster for the first reply, so MTP stays optional and off by default.

All eight MTP answers were reviewed against the input code. The model's earlier bit-width, cache-removal, and generated launch-size errors were absent. The code helper had the required local definitions. All replies stopped at 512 tokens. The follow-up still continued incomplete code instead of giving the requested test explanation. The generated program was not executed. These two tasks do not establish quality for other tasks.

Version 0.1.1 passed 669 tests and 23 subtests; nine tests were skipped because private fixtures are absent. Both installed environments matched all 45 tested runtime files. Model files and dependency versions were unchanged.

Native affine experts increased normal decode speed by 15.4% on the first reply and 15.8% on the follow-up. The new test used one production model load. It rebuilt the 23,860 cached prompt tokens with the new expert values. Each code reply used 512 output tokens. The follow-up added the same 4,469 input tokens. The reference runs were earlier; these comparisons were not paired.

| Mode | First reply tokens/s | Follow-up tokens/s | Total follow-up seconds |
|---|---:|---:|---:|
| v0.1.1 normal | 13.78 | 13.67 | 78.22 |
| Native affine normal | 15.90 | 15.83 | 67.72 |
| v0.1.1 MTP, draft limit three | 12.05 | 18.10 | 69.64 |
| Native affine MTP, draft limit three | 14.85 | 23.73 | 57.43 |

Against the earlier MTP medians, first-reply speed increased by 23.2% and follow-up speed increased by 31.1%. Normal generation remains faster for the first reply. MTP remains off by default.

Peak memory was 429.82 GiB. This process held the K3 draft layer and MTP history during normal requests. Main expert storage increased by 12.5%. Loading and conversion took 165.50 seconds; fresh prefix preparation took 163.73 seconds. Both times are outside the request measurements. The earlier MTP load took 88.63 seconds, but load times were not measured as a controlled pair.

The normal replies stayed close to the reference answers. The bit widths, shared expert offset, cache admission, and generated fused launch size were correct. The first reply also made an unverified cache-memory estimate. The MTP follow-up used a wrong generated launch size: `features // 288` instead of `features // 128 * 288`. Both calculation answers were correct. The code replies remain incomplete at 512 tokens. The follow-ups continued input code instead of explaining tests. The generated program was not executed. These samples do not establish general model quality.

A separate float16 variant saved about 21 GiB and added about 1.3% to normal decode speed. Both first replies added false 1-bit claims. The normal first reply also added a false cache-loop failure. Its MTP follow-up used the correct launch size. The selected format keeps float32 matrix operands, scales, biases, and results. These observed first-reply errors weighed against the small extra speed.

The selected release passed 693 tests and 23 subtests with Metal enabled. Nine private-fixture tests were skipped. Both installed environments match all 46 tested runtime files. All 236 model-file identities and dependency versions stayed unchanged.

Version 0.1.3 keeps the four-bit expert format and fixes the MTP cost comparison and fallback. The earlier controller used one synchronous target call to measure ordinary generation. If it stopped drafting, it kept synchronous calls for every remaining token. The new controller measures the ordinary MLX-LM pipeline and uses that pipeline after costly draft verification.

| MTP, draft limit three | Previous tokens/s | New tokens/s | Speed change |
|---|---:|---:|---:|
| First reply | 14.83 | 15.74 | +6.1% |
| Follow-up | 23.81 | 23.64 | -0.7% |

This test used the same saved affine4 prefix and canonical follow-up history. Only MTP generation changed. The main operators and MTP cache classes matched the saved prefix. Normal generation was not rerun. Each reply used 512 tokens. Warm-up and compilation were outside timing. The reference run was earlier; this was not a paired comparison. Combined request time fell by 2.0%, from 91.90 to 90.05 seconds. Peak memory was 429.65 GiB.

The first answer kept the main code facts, but added an unsupported cache-accounting claim about padded slices. The follow-up used the correct fused launch size and a valid finish helper. It matched the earlier normal answer. Both answers stopped at 512 tokens; the follow-up still continued input code instead of explaining tests. The generated program was not executed. The tests cover cache state when generation stops at its length limit, reaches EOS, or the client closes the stream.

A three-bit expert trial increased normal speed by about 7% and reduced peak memory by 83.48 GiB. It also added a generated helper syntax error and false weight and cache claims. The user selected four-bit weights with the MTP fix. A seven-token draft limit reduced speed on both tested requests. A float16 dense trial was slower in 11 of 12 component cases. These variants are not enabled.

Version 0.1.3 passed 703 tests and 23 subtests with Metal enabled. Nine private-fixture tests were skipped. Both installed environments match all 46 tested runtime files. Dependency and CLI checks passed.

Version 0.1.4 adds verified prompt lookup to `--mtp`. It proposes matching spans from the prior prompt and generated tokens, then verifies each token with all 78 main layers. It can continue the matching source span and return from ordinary generation when a new useful match appears. The prompt proposal maximum is 63; K3 proposals use at most three. Add `--mtp` to a `generate` or `serve` command to enable this path. MTP stays off by default.

| Request | Earlier MTP tokens/s | New tokens/s | Speed change |
|---|---:|---:|---:|
| Review | 15.74 | 15.32 | -2.7% |
| Code follow-up | 23.64 | 59.19 | +150.4% |

Each reply generated 512 tokens from the same saved four-bit prefix and canonical history. The full runtime test includes the follow-up's 4,469 new input tokens in prompt and total request timing. Prompt work took 35.94 seconds; generation took 8.63 seconds. Total follow-up time fell from 57.49 to 44.57 seconds. Combined first-reply and follow-up time fell by 13.3%. Loading, compilation, warmup, hashes, and finite checks were outside request timing. The earlier runs were separate; normal generation was not rerun.

All 496 code-follow-up proposals were accepted in eight verification passes. The output matches the earlier normal reply, with a valid finish helper and the correct fused launch size. The review's main code facts stay close, but its I/O-dominance claim is unmeasured by this resident test. The two arithmetic answers are correct. Both code replies still stop at 512 tokens, and the follow-up continues source code instead of giving the requested test explanation. The generated program was not executed. The 59.19 tokens/s result applies to this code continuation with useful source matches; the review speed remains 15.32 tokens/s.

Peak MLX allocation was 429.60 GiB. Four-bit main weights and the original K3 layer are retained. All 236 model-file identities and dependency versions stayed unchanged. The portable build passed 716 tests and 23 subtests; nine private-fixture tests were skipped. Tests cover accepted and rejected prompt spans, cached token history, and length, EOS, and client stops within a verified span. See [prompt-lookup results](docs/prompt-lookup-benchmark-results.json).

Version 0.1.5 checks a short prompt span before paying for a long verification pass. This reduces the cost of a poor copy match. K3 remains limited to three proposals.

| Request | v0.1.4 median tokens/s | Short-probe median tokens/s | Speed change |
|---|---:|---:|---:|
| Review | 15.32 | 15.72 | +2.6% |
| Code follow-up | 59.41 | 58.74 | -1.1% |

The paired comparison used long/probe/probe/long order on one resident four-bit model. Each arm produced two 512-token answers on both requests. Combined median request time fell from 78.03 to 77.30 seconds, 0.94% less. Follow-up prompt work is inside request timing; loading, compilation, warmup, and checks are outside. Peak MLX allocation was 429.77 GiB.

The review keeps the main source facts but adds a conditional 1 GiB large-layer example that is not established for this input. All code follow-ups keep the same answer, valid finish helpers, and correct fused launch size. Replies still stop at 512 tokens, and the follow-up continues code instead of explaining tests. The generated program was not executed. These results do not establish 50 tokens/s for general answers.

The integrated generator matches the tested candidate. Version 0.1.5 passed 718 tests and 23 subtests with Metal enabled; nine private-fixture tests were skipped. See [prompt-probe results](docs/prompt-probe-benchmark-results.json). The window-cost method was informed by [Tensorfold's planner](https://github.com/ashhart/TensorFold/blob/c126fe16e3f8d9bef4a620af34ff35aec0966af1/zig/src/core/lanes/plan_lanes.zig).

Version 0.1.6 adds relaxed acceptance for ordinary K3 draft words. The main model checks each proposed word against its output distribution. Prompt matches still use the target choice. K3 stays at three proposals. The controller measures four neural passes before it stops a costly draft source. Add `--mtp` to enable this path; answers can differ. Four-bit main weights remain.

The first balanced comparison used original/relaxed/relaxed/original order on eight task types:

| Task | Original tokens/s | Relaxed tokens/s | Speed change |
|---|---:|---:|---:|
| Source review | 15.66 | 21.24 | +35.7% |
| Interval function | 23.60 | 24.80 | +5.1% |
| Inventory arithmetic | 20.34 | 21.38 | +5.1% |
| Scheduling | 16.88 | 24.18 | +43.2% |
| Invoice extraction | 25.78 | 25.68 | -0.4% |
| Incident summary | 20.51 | 25.47 | +24.2% |
| Cache explanation | 17.00 | 17.00 | +0.0% |
| Maintenance email | 19.92 | 20.68 | +3.8% |

A second balanced comparison used two samples per arm on the same eight prompts. Four neural cost samples increased cache-explanation speed from 16.98 to 23.29 tokens/s, 37.1% faster. Request time fell from 14.29 to 10.34 seconds. The selected version produced 20.09–25.67 tokens/s across these prompts. These results do not establish 50 tokens/s for general answers. The two comparisons used separate model loads; cross-comparison differences are not paired results.

The generated interval functions passed all 519 behavior cases. Inventory, invoice and selected scheduling answers were correct. Incident and email answers retained the supplied facts. The faster review added false one-bit and 79+ layer claims and remained incomplete at 512 tokens. The selected explanation added the false claim that a first run is always slower. These errors are recorded with the speed gains; exact words are not an acceptance condition.

New-prompt processing is inside request time. Source review uses the saved 23,860-token prefix. Loading, compilation, warmup and checks are outside timing. Peak allocation was 413.85 GiB. Version 0.1.6 passed 728 tests and 23 subtests with Metal enabled; nine private-fixture tests were skipped. See [relaxed draft results](docs/relaxed-draft-benchmark-results.json) for all request measurements and answer checks. The measured private generator was ported directly; the integrated release was not timed in a separate full-model run.

Version 0.1.7 uses joined four-bit MXFP4 main experts and new attention layouts. Gate and up projections share input preparation and one matrix operation. Attention uses joined buffers and token/head layouts within the existing guards. With `--mtp`, accepted prompt spans increase from three proposals to 15, then to 63. Rejection returns to three. The eight-token ordinary probe, three-token K3 limit, existing word acceptance rule, and four neural cost samples are retained. All 78 main layers and the original K3 draft layer remain. MTP stays off by default.

The full-model comparison completed 80 requests across eight task types and five settings. Each setting used two samples per task. Four fresh processes ran in previous/joined/joined/previous order. The joined processes used balanced setting order. The source review used a 23,860-token prefix rebuilt with the new expert values. Other tasks used new prompts. New prompt processing is inside request time. Tokenization, loading, prefix preparation, compilation, warmup, cache cloning, and checks are outside timing.

| Task | v0.1.6 median tokens/s | v0.1.7 median tokens/s | Speed change |
|---|---:|---:|---:|
| Source review | 21.16 | 22.45 | +6.1% |
| Interval function | 24.73 | 24.74 | +0.1% |
| Inventory arithmetic | 21.37 | 23.29 | +9.0% |
| Scheduling | 24.20 | 24.14 | -0.2% |
| Invoice extraction | 25.73 | 25.88 | +0.6% |
| Incident summary | 25.49 | 25.22 | -1.0% |
| Cache explanation | 23.39 | 24.67 | +5.5% |
| Maintenance email | 20.15 | 24.57 | +22.0% |

Weighted decode speed increased from 22.83 to 23.82 tokens/s, 4.34% faster. The sum of median request times fell from 75.38 to 68.63 seconds, 8.96% less. The sum of median output lengths also fell from 1,542 to 1,460 tokens. Output lengths include reasoning and stop tokens. Thus, the request-time reduction does not measure kernel speed alone. The 50 tokens/s target was not met.

Warm request peak MLX allocation fell from 413.82 to 392.01 GiB, 21.81 GiB less. Fresh prefix preparation reached 419.31 GiB. These are whole-process peaks; they do not measure added memory for each change.

All ten generated functions passed 519 behavior checks each. All 30 JSON answers had the required values. All 40 code/JSON format checks passed. Incident summaries and maintenance emails retained the given facts. The selected source review still makes a false one-bit claim and adds wrong cache-loop and full-cache eviction claims. It removes earlier unsupported layer-count and memory-map claims. All source reviews stop at 512 tokens and are incomplete. The selected cache explanation removes the false always-slower claim, but omits the live-process condition and repeated comparisons. Its 177 words fit the 180-word limit. The one-token probe trials used 190 words and exceeded that limit; they were not selected. These results do not show a general answer-quality gain. Exact wording is not an acceptance condition.

See [joined MXFP4 results](docs/joined-mxfp4-benchmark-results.json) for the selected settings, all setting totals, per-task measurements, and quality limits. These measurements cover the tested full-model generator. The integrated lookup controller matches that generator in AST and bytecode. Version 0.1.7 passed 762 tests and 23 subtests with Metal enabled; nine private-fixture tests were skipped. The release integration was not timed in a separate full-model run.

See [MTP pipeline results](docs/mtp-pipeline-benchmark-results.json), [native affine results](docs/native-affine-benchmark-results.json), [production benchmark data](docs/benchmark-results.json), [MTP tuning data](docs/mtp-tuning-benchmark-results.json), and [earlier batched MTP data](docs/batched-mtp-benchmark-results.json). The previous MTP data remain in [the earlier report](docs/mtp-benchmark-results.json). For a new resident benchmark with your own input:

```sh
MLX_MAX_MB_PER_BUFFER=2048 MLX_MAX_OPS_PER_BUFFER=100 python tools/benchmark_runtime.py \
  --model "$MODEL" --modes target --input-lengths 512 \
  --max-tokens 128 --output artifacts/my-benchmark.json
```

Stop the API first. This benchmark loads its own resident model.

## Troubleshooting

| Symptom | Action |
|---|---|
| No compatible MLX wheel | Check macOS 26+, Python 3.11, and native `arm64` Python. |
| Missing model file or manifest error | Check the folder layout and the EXLR8 up1 manifest. |
| Memory allocation failure | Stop other model processes. Use SSD-cache mode or a smaller prompt block. Set limits within physical RAM. |
| API connection refused | Wait for model loading to finish. Check the server terminal and port. |
| Model name rejected | Use the API alias `glm-5.3-exlr8-up1`. |

## Attribution and licenses

| Source | Work used |
|---|---|
| [MLX-LM](https://github.com/ml-explore/mlx-lm) / [oMLX](https://github.com/jundot/omlx) | GLM attention and routing logic; MLX-LM generation, tokenizer, cache, and API support |
| [MLX](https://github.com/ml-explore/mlx) | GPU kernel arithmetic and templates |
| [PonyExl3](https://github.com/beamivalice/PonyExl3) | Expert decode and prompt kernel structure; CPU reference checks |
| [exllamav3](https://github.com/turboderp-org/exllamav3) | Weight packing, codebook, and transforms |
| MLXLR8 / EXLR8 | Loader, integration, and tested kernel changes |

Optional MTP logic also follows [vLLM](https://github.com/vllm-project/vllm). The base model is [Z.AI GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16). Prepared EXLR8 quantization files are credited to [KindlingAI](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4); retain their model card and license.

Original MLXLR8 code is under [Apache 2.0](LICENSE). Upstream code retains its licenses and notices. See [detailed attribution](ATTRIBUTIONS.md), [NOTICE](NOTICE), and [license texts](third_party/licenses). Model weights are a separate download and retain their own GLM-5.3 and quantization license terms.

Version 0.1.8 omits explicit LHS index arrays from both one-row joined expert matrix calls. MLX still supplies default indices. Weight encoding, attention and MTP behavior are unchanged. The same eight prompts were tested twice: 23.84 tokens/s, versus the documented PR #8 result of 23.82 tokens/s. Request peak memory was 392.01 GiB. Code and JSON checks passed; incident and email facts were correct. Source and cache explanation faults remain. The release passed 763 tests and 23 subtests with Metal enabled; nine private-fixture tests were skipped. See [one-row index fix results](docs/one-row-index-fix-results.json).


Version 0.1.9 shares each existing FP8 dense weight across four token rows. It keeps F32 input calculations, the sum order, transforms, scales, and final casts. This path needs no additional weight cache.

The same eight prompts were tested twice against the documented v0.1.8 results:

| Measure | v0.1.8 | v0.1.9 | Change |
|---|---:|---:|---:|
| Decode, tokens/s | 23.84 | 24.85 | +4.21% |
| Fresh-prompt prefill, tokens/s | 85.56 | 85.62 | +0.06% |
| Sum of median request times, seconds | 68.55 | 66.09 | -3.60% |
| Peak allocation, GiB | 392.007 | 392.007 | — |

Prefill is new input tokens divided by time to the first token. The seven fresh prompts contain 73–119 input tokens. The source prompt has 23,860 cached tokens and one new token; it is reported separately. Loading, warmup, and cache preparation are outside these measurements. Decode excludes the first output token.

Every prompt had faster decode. All 16 final answers matched v0.1.8. Two code answers passed 519 behavior cases each, six JSON answers passed, and four incident/email answers retained the required facts. The four existing source-review and cache-explanation failures remain. No new answer-quality failure was found. The release passed 766 tests and 23 subtests with Metal enabled; nine private-fixture tests were skipped. The 50 tokens/s target remains unmet. See [shared FP8 results](docs/shared-fp8-results.json).


Version 0.1.10 reads eight FP8 weights in one packed load. Query projections also process two output columns at a time to reduce temporary storage. The existing four-row path keeps its F32 sum order and needs no additional weight cache.

The same eight prompts were tested twice against the documented v0.1.9 results, with the same command-buffer settings (`MLX_MAX_OPS_PER_BUFFER=100`, `MLX_MAX_MB_PER_BUFFER=2048`):

| Measure | v0.1.9 | v0.1.10 | Change |
|---|---:|---:|---:|
| Decode, tokens/s | 24.85 | 25.09 | +0.99% |
| Fresh-prompt prefill, tokens/s | 85.62 | 85.51 | -0.12% |
| Sum of median request times, seconds | 66.09 | 65.53 | -0.84% |
| Peak allocation, GiB | 392.007 | 392.007 | — |

All eight prompts had faster decode. All 16 final answers matched v0.1.9. Two code answers passed 519 behavior cases each, six JSON answers passed, and four incident/email answers retained the required facts. The four existing source-review and cache-explanation failures remain. No new answer-quality failure was found. The unchanged tests passed 766 cases and 23 subtests with Metal enabled; nine private-fixture tests were skipped. The 50 tokens/s target remains unmet. See [packed FP8 results](docs/packed-fp8-results.json) for both speeds per prompt and the measurement definitions.


### Four-row output MXFP4

Version 0.1.11 shares packed MXFP4 output weights across four input rows. Query projections keep v0.1.10. The conversion retains both F32 side transforms and the BF16 return type, but changes inner weight values. It occurs during loading and leaves checkpoint files unchanged.

The same eight prompts ran twice, in forward and reverse order, against documented v0.1.9 results. No baseline model rerun was made. Both runs used `MLX_MAX_OPS_PER_BUFFER=100` and `MLX_MAX_MB_PER_BUFFER=2048`.

| Measure | v0.1.9 | v0.1.11 | Change |
|---|---:|---:|---:|
| Decode, tokens/s | 24.85 | 26.45 | +6.45% |
| Fresh-prompt prefill, tokens/s | 85.62 | 85.55 | -0.08% |
| Sum of median request times, seconds | 66.09 | 62.60 | -5.29% |
| Peak allocation, GiB | 392.01 | 395.74 | +3.73 GiB |

All eight prompts improved decode rate; seven finished sooner. Email time rose 2.28%. Against documented v0.1.10, decode improved 5.41% and summed request time fell 4.48%.

Answer quality is mixed. Both code answers passed 519 behavior cases each; all six JSON answers and four incident/email answers passed. Both versions have 12 passing and four failing task answers. Source reviews remove the false one-bit description but retain unproved cache-accounting claims and add an unsupported model-size claim. Cache explanations add a repeated warmed comparison, but use 187 words against the requested 180; v0.1.9 used 177. Data-lifetime and cache-reset flaws remain. These eight prompts do not establish broad model quality.

The change is retained for the shorter request time and reasonably close answers. The Metal suite passed 779 tests and 23 subtests; nine optional-fixture tests were skipped. All 75 output projections were observed during checked warmup, and 51 runtime files matched the tested source. The 50 tokens/s target remains unmet. See [output MXFP4 results](docs/output-mxfp4-results.json) for prefill and decode per prompt.
