"""Generate with MLX-LM or serve its local OpenAI-compatible API."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_tokenizer, maybe_set_recommended_wired_limit

from .experts import ExpertStore
from .model import load_model
from .mtp import MTPRuntime, stream_generate


DEFAULT_MODEL = Path.home() / ".lmstudio/models/kindlingai/glm-5.3-exlr8-up1"


def load_runtime(args):
    maybe_set_recommended_wired_limit()
    mx.set_cache_limit(1024**3)
    mx.set_memory_limit(int(args.memory_limit_gib * 1024**3))
    start = time.perf_counter()
    optimize = not getattr(args, "no_optimize", False)
    store = ExpertStore(
        args.model, args.expert_cache_gib, resident=getattr(args, "resident", False),
        optimize=optimize,
    )
    def progress(done, total):
        if done % 10 == 0 or done == total:
            print(f"Loaded dense layers: {done}/{total}", file=sys.stderr, flush=True)
    model = load_model(args.model, store, progress=progress, optimize=optimize,
                       indexed_prefill=getattr(args, "indexed_prefill", True),
                       decode_k4=optimize and store.resident)
    config = json.loads((Path(args.model) / "config.json").read_text())
    tokenizer = load_tokenizer(Path(args.model), eos_token_ids=config["eos_token_id"])
    mx.eval(model.parameters())
    if getattr(args, "mtp", False):
        model = MTPRuntime(model, args.model, store, optimize=optimize)
        mx.eval(model.parameters())
    print(json.dumps({
        "loaded_seconds": round(time.perf_counter() - start, 2),
        "active_gib": round(mx.get_active_memory() / 1024**3, 2),
        "expert_cache_limit_gib": args.expert_cache_gib,
        "resident": getattr(args, "resident", False),
        "mtp": getattr(args, "mtp", False),
    }), file=sys.stderr, flush=True)
    if getattr(args, "trace", False):
        previous = time.perf_counter()
        def trace(layer, hidden):
            nonlocal previous
            mx.eval(hidden)
            current = time.perf_counter()
            print(json.dumps({
                "layer": layer, "seconds": round(current - previous, 3),
                "finite": bool(mx.all(mx.isfinite(hidden)).item()),
                "rms": float(mx.sqrt(mx.mean(hidden.astype(mx.float32)**2)).item()),
            }), file=sys.stderr, flush=True)
            previous = current
        (model.target if isinstance(model, MTPRuntime) else model).trace = trace
    return model, tokenizer, store


def generate(args):
    model, tokenizer, store = load_runtime(args)
    prompt = args.prompt
    if not args.raw:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    last = None
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens,
        sampler=make_sampler(args.temp), prefill_step_size=args.prefill_step_size,
        num_draft_tokens=args.num_draft_tokens,
    ):
        print(response.text, end="", flush=True)
        last = response
    print()
    if last is not None:
        print(json.dumps({
            "prompt_tokens": last.prompt_tokens,
            "prompt_tps": round(last.prompt_tps, 3),
            "generation_tokens": last.generation_tokens,
            "generation_tps": round(last.generation_tps, 3),
            "peak_gib": round(mx.get_peak_memory() / 1024**3, 2),
            "finish_reason": last.finish_reason,
            **store.stats(),
            **getattr(model, "_mtp_stats", {}),
        }), file=sys.stderr)


def serve(args, runtime=None):
    from mlx_lm import server as mlx_server
    from mlx_lm.server import ModelProvider, run

    args.adapter_path = args.draft_model = None
    args.pipeline = args.trust_remote_code = args.use_default_chat_template = False
    args.chat_template = ""
    args.chat_template_args = {}
    args.allowed_origins = ["*"]
    args.top_p, args.top_k, args.min_p = 1.0, 0, 0.0
    args.decode_concurrency = args.prompt_concurrency = 1
    args.prompt_cache_size = 2
    args.prompt_cache_bytes = None
    args.kv_bits = None
    args.kv_group_size, args.quantized_kv_start = 64, 5000
    args.num_draft_tokens = getattr(args, "num_draft_tokens", 3)

    class Provider(ModelProvider):
        def __init__(self):
            super().__init__(args)
            self._model_map["glm-5.3-exlr8-up1"] = args.model

        def _load(self, model_path, adapter_path=None, draft_model_path=None):
            if (Path(model_path).resolve() != Path(args.model).resolve()
                    or adapter_path is not None or draft_model_path is not None):
                raise ValueError("This server serves only the selected EXLR8 model")
            self.model, self.tokenizer, self.expert_store = runtime if runtime is not None else load_runtime(args)
            self.model_key = (model_path, None, None)
            self.draft_model = None
            self.is_batchable = False

    logging.basicConfig(level=logging.INFO)
    provider = Provider()
    provider.load_default()
    previous_generate = mlx_server.stream_generate
    mlx_server.stream_generate = stream_generate
    try:
        run(args.host, args.port, provider)
    finally:
        mlx_server.stream_generate = previous_generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "serve"):
        p = sub.add_parser(command)
        p.add_argument("--model", default=str(DEFAULT_MODEL))
        p.add_argument("--expert-cache-gib", type=float, default=32)
        p.add_argument("--memory-limit-gib", type=float)
        p.add_argument("--resident", action="store_true", help="Keep all main experts in RAM")
        p.add_argument("--no-optimize", action="store_true", help="Use packed dense weights and reference expert kernels")
        p.add_argument("--mtp", action="store_true", help="Use the retained K3 MTP layer for verified token proposals")
        p.add_argument("--num-draft-tokens", type=int, default=3)
        p.add_argument("--prefill-step-size", type=int, default=2048)
        p.add_argument("--max-tokens", type=int, default=128)
        p.add_argument("--temp", type=float, default=0)
        if command == "generate":
            p.add_argument("--prompt", required=True)
            p.add_argument("--raw", action="store_true")
            p.add_argument("--trace", action="store_true")
        else:
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    if args.memory_limit_gib is None:
        resident_default = 416 if args.no_optimize else 440
        args.memory_limit_gib = resident_default if args.resident else 96
    if args.expert_cache_gib < 0 or args.memory_limit_gib <= 0:
        parser.error("Memory limits must be positive; expert cache can be zero")
    if args.prefill_step_size <= 0 or args.max_tokens <= 0:
        parser.error("Token counts must be positive")
    if args.num_draft_tokens < 0:
        parser.error("The draft token count must be nonnegative")
    if args.expert_cache_gib >= args.memory_limit_gib - 16:
        parser.error("Leave at least 16 GiB for dense weights and temporary arrays")
    resident_minimum = 384 if args.no_optimize else 416
    if args.resident and args.memory_limit_gib < resident_minimum:
        parser.error(f"Resident mode requires at least {resident_minimum} GiB for the memory limit")
    (generate if args.command == "generate" else serve)(args)
