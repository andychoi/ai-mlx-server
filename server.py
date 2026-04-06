#!/usr/bin/env python3
"""MLX inference server — extends mlx_lm.server with /v1/embeddings.

Runs on Apple Silicon. Exposes an OpenAI-compatible API so any client
that speaks to Ollama or OpenAI can use this server transparently.

Usage:
    python server.py --port 8085
    python server.py --model mlx-community/gemma-4-27b-it-4bit --port 8085
    python server.py --model mlx-community/gemma-4-27b-it-4bit --adapter-path lora/adapters/my-adapter --port 8085

Docker containers reach it via:
    OLLAMA_URL=http://host.docker.internal:8085
"""

import json
import logging
import os
import threading
from functools import partial
from http.server import HTTPServer

from mlx_lm.server import APIHandler, ModelProvider, LRUPromptCache, ResponseGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mlx-server")

# ── Embedding model cache (loaded once per model path) ──────────────────────
_embed_cache: dict[str, tuple] = {}
_embed_lock = threading.Lock()


def _load_embed_model(model_path: str):
    """Load an embedding model, trying offline cache first."""
    from mlx_embeddings import load as mlx_embed_load

    orig = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return mlx_embed_load(model_path)
    except Exception:
        log.info("Embedding model %s not cached, downloading…", model_path)
        if orig is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = orig
        return mlx_embed_load(model_path)
    finally:
        if orig is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = orig


def _get_embed_model(model_path: str):
    """Return cached (model, tokenizer) for the given embedding model."""
    with _embed_lock:
        if model_path not in _embed_cache:
            log.info("Loading embedding model: %s", model_path)
            model, tokenizer = _load_embed_model(model_path)
            _embed_cache[model_path] = (model, tokenizer)
        return _embed_cache[model_path]


# ── Extended API handler ────────────────────────────────────────────────────

class MLXAPIHandler(APIHandler):
    """APIHandler with /v1/embeddings and /health endpoints."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
            return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self):
        log.info("POST %s", self.path)
        if self.path == "/v1/embeddings":
            self._handle_embeddings()
            return
        # Delegate chat/completions to parent
        super().do_POST()

    def _handle_embeddings(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length)
            body = json.loads(raw.decode())

            model_path = body.get("model", "")
            input_text = body.get("input", "")

            if not model_path:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "model is required"}).encode())
                return

            if not input_text:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "input is required"}).encode())
                return

            model, tokenizer = _get_embed_model(model_path)

            with _embed_lock:
                input_ids = tokenizer.encode(input_text, return_tensors="mlx")
                outputs = model(input_ids)
                embedding = outputs.text_embeds[0].tolist()

            prompt_tokens = len(input_text) // 4

            response = {
                "object": "list",
                "data": [{"object": "embedding", "embedding": embedding, "index": 0}],
                "model": model_path,
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            log.error("Embedding failed: %s", e)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX inference server with embeddings and LoRA")
    parser.add_argument("--model", type=str, default=None,
                        help="Default model to preload (HuggingFace path)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8085,
                        help="Port (default: 8085)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code for tokenizer")
    parser.add_argument("--chat-template", type=str, default=None)
    parser.add_argument("--use-default-chat-template", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # mlx_lm.server arguments that ModelProvider/ResponseGenerator need
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter weights")
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--num-draft-tokens", type=int, default=3)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--chat-template-args", type=json.loads, default="{}")
    parser.add_argument("--decode-concurrency", type=int, default=32)
    parser.add_argument("--prompt-concurrency", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--prompt-cache-size", type=int, default=None)
    parser.add_argument("--prompt-cache-bytes", type=int, default=None)
    parser.add_argument("--pipeline", action="store_true")

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    model_provider = ModelProvider(args)
    prompt_cache = LRUPromptCache(args.prompt_cache_size or 1)
    response_generator = ResponseGenerator(model_provider, prompt_cache)

    handler = partial(
        MLXAPIHandler,
        response_generator,
        system_fingerprint="mlx-server",
    )

    server = HTTPServer((args.host, args.port), handler)
    log.info("MLX server listening on %s:%d", args.host, args.port)
    if args.model:
        log.info("Default model: %s", args.model)
    if args.adapter_path:
        log.info("LoRA adapter: %s", args.adapter_path)
    log.info("Endpoints: /v1/chat/completions, /v1/embeddings, /health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down…")
        response_generator.stop_and_join()
        server.server_close()


if __name__ == "__main__":
    main()
