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

import io
import json
import logging
import os
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import HTTPServer

from mlx_lm.server import APIHandler, ModelProvider, LRUPromptCache, ResponseGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mlx-server")

# ── Version (read once at import time) ──────────────────────────────────────
_SERVER_VERSION = "0.1.0"
try:
    import tomllib  # Python 3.11+
    _toml_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    with open(_toml_path, "rb") as _f:
        _SERVER_VERSION = tomllib.load(_f)["project"]["version"]
except Exception:
    pass

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


# ── Ollama adapter ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time in ISO-8601 format (Ollama style)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class OllamaAdapter:
    """Translates between Ollama API format and OpenAI API format."""

    @staticmethod
    def translate_request(ollama_body: dict, *, wrap_prompt: bool = False) -> dict:
        """Convert an Ollama request body to an OpenAI-compatible body.

        Args:
            ollama_body: The parsed Ollama request dict.
            wrap_prompt: If True, treat ``ollama_body["prompt"]`` as a user
                         message (for /api/generate).  If False, pass
                         ``messages`` through as-is (for /api/chat).
        """
        openai_body: dict = {}

        # model
        if "model" in ollama_body:
            openai_body["model"] = ollama_body["model"]

        # messages
        if wrap_prompt:
            prompt = ollama_body.get("prompt", "")
            openai_body["messages"] = [{"role": "user", "content": prompt}]
        else:
            if "messages" in ollama_body:
                openai_body["messages"] = ollama_body["messages"]

        # stream — Ollama defaults to True, OpenAI defaults to False
        openai_body["stream"] = ollama_body.get("stream", True)

        # options mapping
        options = ollama_body.get("options", {})
        if "num_predict" in options:
            openai_body["max_tokens"] = options["num_predict"]
        if "temperature" in options:
            openai_body["temperature"] = options["temperature"]
        if "top_p" in options:
            openai_body["top_p"] = options["top_p"]
        if "top_k" in options:
            openai_body["top_k"] = options["top_k"]
        if "stop" in options:
            openai_body["stop"] = options["stop"]

        # pass through stop at top level too
        if "stop" in ollama_body and "stop" not in openai_body:
            openai_body["stop"] = ollama_body["stop"]

        return openai_body

    @staticmethod
    def openai_response_to_ollama(openai_resp: dict, model: str) -> dict:
        """Convert a non-streaming OpenAI response dict to Ollama format."""
        content = ""
        finish_reason = "stop"
        if openai_resp.get("choices"):
            choice = openai_resp["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            finish_reason = choice.get("finish_reason", "stop") or "stop"

        return {
            "model": model,
            "created_at": _now_iso(),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": finish_reason,
        }

    @staticmethod
    def openai_sse_chunk_to_ollama(sse_data: str, model: str) -> dict | None:
        """Parse one SSE data payload and return an Ollama NDJSON chunk dict.

        Returns None if the chunk should be skipped (e.g. ``[DONE]``).
        """
        sse_data = sse_data.strip()
        if sse_data == "[DONE]":
            return None
        try:
            chunk = json.loads(sse_data)
        except json.JSONDecodeError:
            return None

        content = ""
        finish_reason = None
        if chunk.get("choices"):
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            content = delta.get("content", "") or ""
            finish_reason = choice.get("finish_reason")

        done = finish_reason is not None
        ollama_chunk: dict = {
            "model": model,
            "created_at": _now_iso(),
            "message": {"role": "assistant", "content": content},
            "done": done,
        }
        if done:
            ollama_chunk["done_reason"] = finish_reason or "stop"
        return ollama_chunk


# ── Streaming wfile wrapper ──────────────────────────────────────────────────

class _StreamTranslatorWfile:
    """Wraps self.wfile and translates SSE lines to Ollama NDJSON on-the-fly.

    The parent handler writes SSE lines like:
        data: {...}\n\n
        data: [DONE]\n\n

    This wrapper intercepts each write, parses SSE data lines, converts them
    to Ollama NDJSON, and writes them to the real wfile.
    """

    def __init__(self, real_wfile, model: str):
        self._real = real_wfile
        self._model = model
        self._buf = b""

    def write(self, data: bytes) -> int:
        self._buf += data
        self._flush_lines()
        return len(data)

    def _flush_lines(self):
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line_str = line.decode("utf-8", errors="replace").rstrip("\r")
            if line_str.startswith("data: "):
                payload = line_str[len("data: "):]
                ollama_chunk = OllamaAdapter.openai_sse_chunk_to_ollama(
                    payload, self._model
                )
                if ollama_chunk is not None:
                    self._real.write(
                        (json.dumps(ollama_chunk) + "\n").encode("utf-8")
                    )
            # Other SSE lines (comments like ": keepalive ...") are silently dropped.

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


# ── Extended API handler ────────────────────────────────────────────────────

class MLXAPIHandler(APIHandler):
    """APIHandler with /v1/embeddings, /health, and Ollama-compatible endpoints."""

    # Loaded model name is injected by main() after handler construction.
    _default_model: str = ""

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
            return
        if self.path == "/api/version":
            self._json_response(200, {"version": _SERVER_VERSION})
            return
        if self.path == "/api/tags":
            models = []
            if self._default_model:
                models.append({
                    "name": self._default_model,
                    "modified_at": _now_iso(),
                    "size": 0,
                    "details": {},
                })
            self._json_response(200, {"models": models})
            return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self):
        log.info("POST %s", self.path)
        if self.path == "/v1/embeddings":
            self._handle_embeddings()
            return
        if self.path == "/api/embeddings":
            self._handle_ollama_embeddings()
            return
        if self.path == "/api/generate":
            self._handle_ollama_chat(wrap_prompt=True)
            return
        if self.path == "/api/chat":
            self._handle_ollama_chat(wrap_prompt=False)
            return
        # Delegate chat/completions (and /v1/completions) to parent
        super().do_POST()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _json_response(self, status: int, body: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    # ── Ollama route handlers ────────────────────────────────────────────────

    def _handle_ollama_embeddings(self):
        """Handle POST /api/embeddings — Ollama uses 'prompt' instead of 'input'."""
        try:
            raw = self._read_body()
            body = json.loads(raw.decode())

            # Ollama uses "prompt" for the text; remap to "input"
            if "prompt" in body and "input" not in body:
                body["input"] = body.pop("prompt")

            # Re-encode and inject into rfile so _handle_embeddings can read it
            remapped = json.dumps(body).encode()
            self.headers["Content-Length"] = str(len(remapped))
            self.rfile = io.BytesIO(remapped)
            self._handle_embeddings()
        except Exception as e:
            log.error("Ollama embeddings failed: %s", e)
            self._json_response(500, {"error": str(e)})

    def _handle_ollama_chat(self, *, wrap_prompt: bool):
        """Handle POST /api/generate or /api/chat.

        Translates the Ollama request to OpenAI format, forwards to the parent
        handler, and translates the response back to Ollama format.

        For streaming responses: wraps self.wfile with _StreamTranslatorWfile
        so SSE output is converted to NDJSON on-the-fly.

        For non-streaming responses: buffers the OpenAI response, then converts
        and writes the Ollama JSON to the real wfile.
        """
        try:
            raw = self._read_body()
            ollama_body = json.loads(raw.decode())
        except Exception as e:
            log.error("Failed to parse Ollama request: %s", e)
            self._json_response(400, {"error": f"Invalid JSON: {e}"})
            return

        model = ollama_body.get("model", self._default_model)
        openai_body = OllamaAdapter.translate_request(ollama_body, wrap_prompt=wrap_prompt)
        is_streaming = openai_body.get("stream", True)

        # Prepare to call parent's do_POST with remapped path + body
        encoded_body = json.dumps(openai_body).encode()
        original_path = self.path
        original_rfile = self.rfile
        original_wfile = self.wfile

        self.path = "/v1/chat/completions"
        self.headers["Content-Length"] = str(len(encoded_body))
        self.rfile = io.BytesIO(encoded_body)

        try:
            if is_streaming:
                # Intercept wfile to translate SSE → NDJSON
                self.wfile = _StreamTranslatorWfile(original_wfile, model)
                super().do_POST()
                # Flush any remaining buffered bytes
                self.wfile.flush()
            else:
                # Buffer the entire OpenAI response, then translate
                buf = io.BytesIO()
                self.wfile = buf
                super().do_POST()

                buf.seek(0)
                raw_response = buf.read()

                # The parent sends HTTP headers + body into self.wfile together.
                # We need to split headers from body. Look for the blank line.
                if b"\r\n\r\n" in raw_response:
                    _headers_part, body_part = raw_response.split(b"\r\n\r\n", 1)
                else:
                    body_part = raw_response

                try:
                    openai_resp = json.loads(body_part.decode())
                    ollama_resp = OllamaAdapter.openai_response_to_ollama(openai_resp, model)
                    self.wfile = original_wfile
                    self._json_response(200, ollama_resp)
                except Exception as e:
                    log.error("Response translation failed: %s", e)
                    # Fallback: write the raw OpenAI response
                    self.wfile = original_wfile
                    original_wfile.write(raw_response)
        finally:
            self.path = original_path
            self.rfile = original_rfile
            self.wfile = original_wfile

    # ── /v1/embeddings ───────────────────────────────────────────────────────

    def _handle_embeddings(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length)
            body = json.loads(raw.decode())

            model_path = body.get("model", "")
            input_value = body.get("input", "")

            if not model_path:
                self._json_response(400, {"error": "model is required"})
                return

            # Normalise to a list; reject empty input
            if isinstance(input_value, list):
                texts = input_value
            else:
                texts = [input_value]

            if not texts or not any(texts):
                self._json_response(400, {"error": "input is required"})
                return

            model, tokenizer = _get_embed_model(model_path)

            data = []
            prompt_tokens = 0

            with _embed_lock:
                for i, text in enumerate(texts):
                    input_ids = tokenizer.encode(text, return_tensors="mlx")
                    outputs = model(input_ids)
                    embedding = outputs.text_embeds[0].tolist()
                    # Determine token count from the encoded result.
                    # With return_tensors="mlx" the result is 2-D (1, seq_len);
                    # fall back to the raw list length for plain list returns.
                    try:
                        token_count = input_ids.shape[-1]
                    except AttributeError:
                        token_count = len(input_ids)
                    prompt_tokens += token_count
                    data.append({"object": "embedding", "embedding": embedding, "index": i})

            response = {
                "object": "list",
                "data": data,
                "model": model_path,
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }
            self._json_response(200, response)

        except Exception as e:
            log.error("Embedding failed: %s", e)
            self._json_response(500, {"error": str(e)})


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

    # Inject the default model name into the handler class so /api/tags can report it.
    MLXAPIHandler._default_model = args.model or ""

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
    log.info(
        "Endpoints: /v1/chat/completions, /v1/embeddings, /health, "
        "/api/generate, /api/chat, /api/embeddings, /api/tags, /api/version"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down…")
        response_generator.stop_and_join()
        server.server_close()


if __name__ == "__main__":
    main()
