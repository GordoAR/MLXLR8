"""Exercise the real MLX-LM HTTP path with a small synthetic model."""

import argparse
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from exlr8.cli import serve


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = mx.array([-10, -10, -10, 10, -10], dtype=mx.float32)
        self.cache_calls = 0
        self.offsets = []

    def make_cache(self):
        self.cache_calls += 1
        return [CacheList(KVCache(), KVCache()), CacheList(KVCache())]

    def __call__(self, inputs, cache=None):
        self.offsets.append(cache[0][0].offset)
        values = inputs.astype(mx.float32)[:, None, :, None]
        for index, group in enumerate(cache):
            for subindex, item in enumerate(group.caches):
                # Match the indexer's cache with an empty value dimension.
                v = mx.zeros(values.shape[:-1] + (0,)) if index == 0 and subindex == 1 else values
                item.update_and_fetch(values, v)
        return mx.broadcast_to(self.logits, inputs.shape + (self.logits.size,))


def tiny_tokenizer():
    tokenizer = Tokenizer(WordLevel(
        {"<unk>": 0, "<eos>": 1, "hello": 2, "ok": 3, "world": 4},
        unk_token="<unk>",
    ))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>",
        chat_template="{{ messages | map(attribute='content') | join(' ') }}",
    )
    return TokenizerWrapper(fast)


class ServerTests(unittest.TestCase):
    def test_http_generation_aliases_streaming_and_prefix_cache(self):
        from mlx_lm import server as mlx_server

        model = TinyModel()
        tokenizer = tiny_tokenizer()
        mx.eval(model.parameters())

        def run_http(host, port, generator):
            provider = generator.model_provider
            httpd = ThreadingHTTPServer((host, 0), lambda *a, **kw: mlx_server.APIHandler(generator, *a, **kw))
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            url = f"http://{host}:{httpd.server_port}"

            def request(route, body=None):
                data = None if body is None else json.dumps(body).encode()
                req = urllib.request.Request(url + route, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    return response.read().decode()

            try:
                self.assertFalse(provider.is_batchable)
                self.assertIsNone(provider.draft_model)
                self.assertEqual(provider.load("default_model"), provider.load("glm-5.3-exlr8-up1"))
                with self.assertRaises(ValueError):
                    provider.load("another-model")
                self.assertEqual(json.loads(request("/health")), {"status": "ok"})
                models = json.loads(request("/v1/models"))
                self.assertIn(str(Path(provider.cli_args.model).resolve()), [entry["id"] for entry in models["data"]])
                prompt = "hello hello hello hello hello hello"
                first = json.loads(request("/v1/completions", {"prompt": prompt, "max_tokens": 2}))
                self.assertEqual(first["choices"][0]["text"], "ok ok")
                self.assertEqual(first["choices"][0]["finish_reason"], "length")
                self.assertEqual(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0)

                before_reuse = len(model.offsets)
                second = json.loads(request("/v1/completions", {
                    "model": "glm-5.3-exlr8-up1", "prompt": prompt, "max_tokens": 2,
                }))
                self.assertEqual(second["choices"][0]["text"], "ok ok")
                self.assertGreater(second["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
                self.assertEqual(model.cache_calls, 1)
                self.assertEqual(model.offsets[before_reuse], second["usage"]["prompt_tokens_details"]["cached_tokens"])
                before_extension = len(model.offsets)
                extension = json.loads(request("/v1/completions", {
                    "prompt": prompt + " world", "max_tokens": 2,
                }))
                self.assertEqual(extension["choices"][0]["text"], "ok ok")
                self.assertEqual(extension["usage"]["prompt_tokens_details"]["cached_tokens"], 6)
                self.assertEqual(model.offsets[before_extension], 6)

                chat = json.loads(request("/v1/chat/completions", {
                    "model": "glm-5.3-exlr8-up1", "messages": [{"role": "user", "content": "hello world"}],
                    "max_tokens": 2,
                }))
                self.assertEqual(chat["choices"][0]["message"]["content"], "ok ok")
                streamed = request("/v1/chat/completions", {
                    "messages": [{"role": "user", "content": "hello world"}], "stream": True,
                    "max_tokens": 2, "stream_options": {"include_usage": True},
                })
                self.assertIn("data: [DONE]", streamed)
                events = [json.loads(line[6:]) for line in streamed.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                text = "".join(choice.get("delta", {}).get("content", "")
                               for event in events for choice in event["choices"])
                self.assertEqual(text, "ok ok")
                self.assertTrue(any("usage" in event for event in events))
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)
                generator.stop_and_join()

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(model=tmp, host="127.0.0.1", port=0,
                                      max_tokens=2, temp=0, prefill_step_size=4)
            with patch("exlr8.cli.load_runtime", return_value=(model, tokenizer, object())) as load:
                with patch("mlx_lm.server._run_http_server", side_effect=run_http):
                    serve(args)
                self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
