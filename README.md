# MLXLR8

Run GLM-5.3 EXLR8 **up1** weights on Apple Silicon with MLX. MLXLR8 adds the EXLR8 loader, MLX integration, and tested kernel changes. The Python package and terminal command are named `exlr8`.

The runtime keeps expert weights packed, applies the format's required E4M3 rounding, and serves a local OpenAI-compatible chat API. All 78 main layers run. MTP is off by default.

## Requirements

- Apple Silicon Mac with **macOS 26 or later**. The pinned MLX wheels require this version.
- Native **Python 3.11** and Git. Check that Python reports `arm64`.
- A complete local **GLM-5.3 EXLR8 up1** model folder from [KindlingAI's EXLR8 checkpoint](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4), with the files below.
- For the tested resident profile: **M3 Ultra with 512 GB unified memory**. The measured peak was 385.26 GiB. Leave memory for macOS and other apps.
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
  --model "$MODEL" --resident --memory-limit-gib 416 \
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
  --model "$MODEL" --resident --memory-limit-gib 416 \
  --host 127.0.0.1 --port 8081
```

For SSD-cache mode, replace `--resident --memory-limit-gib 416` with `--expert-cache-gib 32 --memory-limit-gib 96 --prefill-step-size 512`.

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

Resident mode enables the retained dense K4 path, including with `--mtp`. Dense K4 projections, residual-add/RMS fusion, and attention lookup support one to four BF16 token rows in a batch of one. RMS fusion uses the verified M3 Ultra, 6,144-channel profile. Larger token batches use the existing operators. `--no-optimize` selects the reference operators for diagnosis. MTP retains the production settings and uses the production prefill input sizes. The requested draft count is a maximum. MTP measures cost and can stop proposals. One to four target rows use the shared dense, compiled expert finish, and indexed sparse attention paths. MTP remains off by default.

## Checks and measured results

Install the test extra and run the portable suite from the repository folder:

```sh
python -m pip install -c requirements-macos-arm64.txt '.[test]'
EXLR8_TEST_METAL=1 python -m pytest -q
```

The repository includes PonyExl3's small CPU reference files for these tests. Tests that need private captured tensors skip when those fixtures are absent. No model download is needed for the portable tests.

The current code passed 621 tests and 23 subtests with Metal tests enabled. Nine tests were skipped. The clean release package also passed normal and streamed chat requests with the local up1 model.

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

The portable release passed 669 tests and 23 subtests; nine tests were skipped because private fixtures are absent. Both installed environments match all 45 tested runtime files. Model files and dependency versions are unchanged.

See [production benchmark data](docs/benchmark-results.json), [MTP tuning data](docs/mtp-tuning-benchmark-results.json), and [earlier batched MTP data](docs/batched-mtp-benchmark-results.json). The previous MTP data remain in [the earlier report](docs/mtp-benchmark-results.json). For a new resident benchmark with your own input:

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
