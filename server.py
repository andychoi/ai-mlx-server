#!/usr/bin/env python3
"""MLX inference server — extends mlx_lm.server with /v1/embeddings.

Runs on Apple Silicon. Exposes an OpenAI-compatible API so any client
that speaks to Ollama or OpenAI can use this server transparently.

Usage:
    python server.py
    python server.py --model mlx-community/gemma-4-27b-it-4bit
    python server.py --model mlx-community/gemma-4-27b-it-4bit --adapter-path lora/adapters/my-adapter
    python server.py --list   # show locally cached HuggingFace models and exit

Docker containers reach it via:
    OLLAMA_URL=http://host.docker.internal:11434
"""

import io
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from functools import partial
import socket
from http.server import HTTPServer


class DualStackHTTPServer(HTTPServer):
    """HTTPServer that binds to both IPv4 and IPv6.

    httpx (used by the ollama SDK) applies Happy Eyeballs and prefers IPv6.
    Python's default HTTPServer only binds AF_INET, so IPv6 attempts are refused.
    Setting IPV6_V6ONLY=False on an AF_INET6 socket creates a dual-stack listener
    that accepts both native IPv6 and IPv4-mapped (::ffff:x.x.x.x) connections.
    """
    address_family = socket.AF_INET6

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()

from mlx_lm.server import APIHandler, ModelProvider, LRUPromptCache, ResponseGenerator

from cache import ModelCache, estimate_model_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mlx-server")

# ── Server start time (for uptime reporting) ─────────────────────────────────
_server_start_time = time.time()

# ── Optional Prometheus metrics ──────────────────────────────────────────────
try:
    from metrics import requests_total, request_duration_seconds, resident_models, resident_bytes, queue_depth as _queue_depth_gauge
    _metrics_enabled = True
except ImportError:
    _metrics_enabled = False

# ── Version (read once at import time) ──────────────────────────────────────
_SERVER_VERSION = "0.1.0"
try:
    import tomllib  # Python 3.11+
    _toml_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    with open(_toml_path, "rb") as _f:
        _SERVER_VERSION = tomllib.load(_f)["project"]["version"]
except Exception:
    pass

# ── Unified model cache ──────────────────────────────────────────────────────
_model_cache = ModelCache()
# Lock for the actual embedding model load (cache lock is only for dict ops)
_embed_lock = threading.Lock()
# Track all loaded model names for /api/tags
_loaded_model_names: set[str] = set()

# ── Per-model ResponseGenerator worker registry ─────────────────────────────
_model_workers: dict[str, "ResponseGenerator"] = {}
_workers_lock = threading.Lock()
# Set by main() so _get_or_create_worker can build ModelProvider instances.
_server_args = None

# ── Model alias table (populated from models.yaml aliases: section) ──────────
_model_aliases: dict[str, str] = {}


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


def _resolve_model_path(model_path: str) -> str:
    """Resolve a model name to a HuggingFace repo ID.

    Resolution order:
    1. Exact alias match (e.g. 'gemma4:e4b' → configured HF repo)
    2. Tag-stripped alias match (e.g. 'gemma4:e4b' strips to 'gemma4', checks alias)
    3. Strip Ollama-style :tag suffix (HuggingFace IDs don't allow colons)
    4. Bare names get 'mlx-community/' prefix (e.g. 'gemma4' → 'mlx-community/gemma4')
    5. Local paths ('/' or '.') are returned unchanged.
    """
    if not model_path:
        return model_path
    # 1. Exact alias lookup (tag-aware: 'gemma4:e4b' → specific repo)
    if model_path in _model_aliases:
        return _model_aliases[model_path]
    # 2. Tag-stripped alias lookup, then general tag stripping
    if ":" in model_path and not model_path.startswith("/") and not model_path.startswith("."):
        stripped = model_path.split(":")[0]
        if stripped in _model_aliases:
            return _model_aliases[stripped]
        model_path = stripped
    # 3. Bare name → mlx-community namespace
    if "/" not in model_path and not model_path.startswith("."):
        return f"mlx-community/{model_path}"
    return model_path


def _get_embed_model(model_path: str):
    """Return cached (model, tokenizer) for the given embedding model."""
    model_path = _resolve_model_path(model_path)
    cached = _model_cache.get(model_path)
    if cached is not None:
        return cached
    with _embed_lock:
        # Double-check after acquiring lock
        cached = _model_cache.get(model_path)
        if cached is not None:
            return cached
        log.info("Loading embedding model: %s", model_path)
        model, tokenizer = _load_embed_model(model_path)
        est = estimate_model_bytes(model_path)
        _model_cache.put(model_path, (model, tokenizer), est_bytes=est, role="embedding")
        _loaded_model_names.add(model_path)
        return (model, tokenizer)


def _get_or_create_worker(model_id: str) -> "ResponseGenerator":
    """Return the per-model ResponseGenerator, creating it lazily if needed.

    Thread-safe: concurrent calls for the same model create only one worker.
    Uses _server_args to build ModelProvider; must be called after main() sets it.
    """
    with _workers_lock:
        if model_id in _model_workers:
            return _model_workers[model_id]

    # Build outside lock to avoid holding it during potentially slow setup.
    import copy
    import argparse
    import psutil

    # Pre-flight: evict unpinned models if the new model may not fit.
    est = estimate_model_bytes(model_id)
    if est > 0:
        try:
            vm = psutil.virtual_memory()
            # Leave a 2 GB buffer for KV cache and OS overhead.
            headroom = vm.available - 2 * 1024 ** 3
            if est > headroom:
                log.warning(
                    "Model %s needs ~%.1f GB but only ~%.1f GB available — "
                    "evicting unpinned models first.",
                    model_id, est / 1e9, vm.available / 1e9,
                )
                _model_cache.evict_all_unpinned()
        except Exception:
            pass

    base = _server_args if _server_args is not None else argparse.Namespace()
    args_copy = copy.copy(base)
    args_copy.model = _resolve_model_path(model_id)
    # mlx_lm ≥0.31.2 reads cli_args.allowed_origins in _set_cors_headers.
    if not hasattr(args_copy, "allowed_origins"):
        args_copy.allowed_origins = []
    mp = ModelProvider(args_copy)
    pc = LRUPromptCache(getattr(args_copy, "prompt_cache_size", None) or 1)
    worker = ResponseGenerator(mp, pc)

    with _workers_lock:
        # Re-check: another thread may have created one while we were building.
        if model_id not in _model_workers:
            _model_workers[model_id] = worker
        else:
            worker.stop_and_join()
            worker = _model_workers[model_id]

    return worker


def _tear_down_worker(model_id: str) -> None:
    """Stop and remove the worker for model_id.

    Called by the ModelCache on_evict callback. In-flight and queued requests
    complete before the worker thread stops.
    """
    with _workers_lock:
        worker = _model_workers.pop(model_id, None)
    if worker is not None:
        log.info("Tearing down worker for %s", model_id)
        worker.stop_and_join()


def _parse_models_config(path: str) -> tuple[list[dict], dict[str, str]]:
    """Parse a simple models.yaml. No pyyaml required.

    Supports two top-level sections:
      models:   list of {id, role} entries to preload
      aliases:  mapping of short name / Ollama tag → HuggingFace repo ID

    Example aliases section:
      aliases:
        gemma4:e4b: mlx-community/gemma-4-e4b-it-4bit
        gemma4:31b: mlx-community/gemma-4-31b-it-4bit
        gemma4: mlx-community/gemma-4-e4b-it-4bit
    """
    models = []
    aliases: dict[str, str] = {}
    current: dict = {}
    section = None  # 'models' | 'aliases'
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if re.match(r'^models:\s*$', line):
                section = "models"
                continue
            if re.match(r'^aliases:\s*$', line):
                section = "aliases"
                continue
            if section == "models":
                if re.match(r'\s*-\s+id:', line):
                    if current:
                        models.append(current)
                    current = {"id": re.sub(r'\s*-\s+id:\s*', '', line).strip()}
                elif re.match(r'\s+role:', line):
                    current["role"] = line.split(":", 1)[1].strip()
            elif section == "aliases":
                # Split on first ': ' — key may itself contain colons (e.g. 'gemma4:e4b')
                if ": " in line:
                    key, _, val = line.partition(": ")
                    key = key.strip()
                    val = val.strip()
                    if key and val:
                        aliases[key] = val
    if current:
        models.append(current)
    return models, aliases


# ── Ollama adapter ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time in ISO-8601 format (Ollama style)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert a list of OpenAI messages to a single prompt string for invoke()."""
    if not messages:
        return ""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # multimodal content — extract text parts only
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


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

        # thinking control — optional, only forwarded when client sends them
        if "enable_thinking" in options:
            openai_body["enable_thinking"] = options["enable_thinking"]

        # chat_template_kwargs — arbitrary extra kwargs for apply_chat_template
        if "chat_template_kwargs" in ollama_body:
            openai_body["chat_template_kwargs"] = ollama_body["chat_template_kwargs"]

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
        self._headers_passed = False  # True once \r\n\r\n separator forwarded

    def write(self, data: bytes) -> int:
        if not self._headers_passed:
            self._buf += data
            sep = self._buf.find(b"\r\n\r\n")
            if sep == -1:
                # Headers not yet complete — pass through raw so status line
                # and headers reach the client unchanged.
                try:
                    self._real.write(self._buf)
                except BrokenPipeError:
                    pass
                self._buf = b""
            else:
                # Forward everything up to and including the separator, then
                # switch to SSE→NDJSON translation for the body.
                try:
                    self._real.write(self._buf[:sep + 4])
                except BrokenPipeError:
                    pass
                self._buf = self._buf[sep + 4:]
                self._headers_passed = True
                try:
                    self._flush_lines()
                except BrokenPipeError:
                    pass
            return len(data)

        self._buf += data
        try:
            self._flush_lines()
        except BrokenPipeError:
            pass
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
                    try:
                        self._real.write(
                            (json.dumps(ollama_chunk) + "\n").encode("utf-8")
                        )
                    except BrokenPipeError:
                        return
            # Other SSE lines (comments, blank lines) are silently dropped.

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


# ── Extended API handler ────────────────────────────────────────────────────

class MLXAPIHandler(APIHandler):
    """APIHandler with /v1/embeddings, /health, and Ollama-compatible endpoints."""

    # Loaded model name is injected by main() after handler construction.
    _default_model: str = ""
    # ResponseGenerator is injected by main() so /health and /metrics can read queue depth.
    _response_generator = None
    # Auth config — injected by main()
    _api_key: str = ""
    _auth_health: bool = False
    _auth_metrics: bool = False

    def _check_auth(self, path: str) -> bool:
        """Return True if the request is authorized, False otherwise.

        If no API key is configured, always returns True.
        /health and /metrics are exempt unless --auth-health/--auth-metrics is set.
        Uses constant-time comparison to prevent timing attacks.
        """
        import hmac
        if not self._api_key:
            return True

        # Check exemptions
        if path == "/health" and not self._auth_health:
            return True
        if path == "/metrics" and not self._auth_metrics:
            return True

        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False

        provided_key = auth_header[len("Bearer "):]
        return hmac.compare_digest(provided_key, self._api_key)

    def do_GET(self):
        if not self._check_auth(self.path):
            self._json_response(401, {"error": "Unauthorized"})
            return
        if self.path == "/health":
            self._handle_health()
            return
        if self.path == "/metrics":
            self._handle_metrics()
            return
        if self.path == "/api/version":
            self._json_response(200, {"version": _SERVER_VERSION})
            return
        if self.path == "/api/tags":
            names = set(_loaded_model_names)
            if self._default_model:
                names.add(self._default_model)
            models = [
                {
                    "name": name,
                    "modified_at": _now_iso(),
                    "size": 0,
                    "details": {},
                }
                for name in sorted(names)
            ]
            self._json_response(200, {"models": models})
            return
        if self.path == "/api/ps":
            self._handle_ollama_ps()
            return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self):
        if not self._check_auth(self.path):
            self._json_response(401, {"error": "Unauthorized"})
            return
        log.info("POST %s", self.path)
        if self.path == "/v1/embeddings":
            self._handle_embeddings()
            return
        if self.path == "/api/embeddings":
            self._handle_ollama_embeddings()
            return
        if self.path == "/api/embed":
            # Newer Ollama endpoint — accepts {"model":..., "input": str|list}
            # Returns Ollama format: {"embeddings": [[...]]}
            self._handle_ollama_embed()
            return
        if self.path == "/api/show":
            self._handle_ollama_show()
            return
        if self.path == "/api/generate":
            self._handle_ollama_chat(wrap_prompt=True)
            return
        if self.path == "/api/chat":
            self._handle_ollama_chat(wrap_prompt=False)
            return
        if self.path == "/v1/chat/completions":
            # Peek at body for thinking params and model routing (single read).
            body = self._peek_body()
            if body:
                model_id = body.get("model") or self._default_model
                if model_id:
                    log.info("  model=%s", model_id)
            if body and ("thinking_budget" in body or "enable_thinking" in body):
                self._handle_thinking_completion(body)
                return
            # Route to per-model worker when _server_args is available.
            if _server_args is not None and body:
                model_id = body.get("model") or self._default_model
                if model_id:
                    self._dispatch_to_worker(model_id)
                    return
        # Delegate chat/completions (and /v1/completions) to parent
        super().do_POST()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _handle_health(self):
        import psutil
        vm = psutil.virtual_memory()
        ram_used_gb = round(vm.used / 1e9, 2)
        ram_available_gb = round(vm.available / 1e9, 2)

        # Per-model queue depths using the correct ResponseGenerator.requests attribute.
        with _workers_lock:
            workers_snapshot = dict(_model_workers)
        queue_depths = {
            mid: w.requests.qsize()
            for mid, w in workers_snapshot.items()
        }
        # Include the default generator if no per-model workers exist yet.
        if not queue_depths:
            default_rg = getattr(MLXAPIHandler, '_response_generator', None)
            default_q = getattr(default_rg, 'requests', None)
            if default_q is not None:
                queue_depths["_default"] = default_q.qsize()

        # Merge cache stats (preloaded/embedding) with lazy-loaded worker models.
        cache_stats = {m["id"]: m for m in _model_cache.stats()}
        for mid in workers_snapshot:
            if mid not in cache_stats:
                est = estimate_model_bytes(mid)
                cache_stats[mid] = {
                    "id": mid,
                    "role": "chat",
                    "size_gb": round(est / 1e9, 2),
                    "loaded_at": None,
                    "last_used_at": None,
                    "calls": None,
                    "pinned": False,
                }
        resident_models = list(cache_stats.values())

        self._json_response(200, {
            "status": "ok",
            "version": _SERVER_VERSION,
            "uptime_seconds": int(time.time() - _server_start_time),
            "resident_models": resident_models,
            "ram_used_gb": ram_used_gb,
            "ram_available_gb": ram_available_gb,
            "queue_depths": queue_depths,
        })

    def _handle_metrics(self):
        if not _metrics_enabled:
            self._json_response(503, {"error": "prometheus_client not installed. pip install ai-mlx-server[metrics]"})
            return
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        # Update gauges from current cache state
        resident_models.set(len(_model_cache))
        resident_bytes.set(_model_cache.total_bytes())
        # Sum queue depths across all per-model workers.
        with _workers_lock:
            workers_snapshot = dict(_model_workers)
        total_queued = sum(w.requests.qsize() for w in workers_snapshot.values())
        if not workers_snapshot:
            default_rg = getattr(MLXAPIHandler, '_response_generator', None)
            default_q = getattr(default_rg, 'requests', None)
            if default_q is not None:
                total_queued += default_q.qsize()
        _queue_depth_gauge.set(total_queued)
        body = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, status: int, body: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    def _peek_body(self) -> dict | None:
        """Read the request body and replace self.rfile so parent can re-read it."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length)
            self.rfile = io.BytesIO(raw)  # restore for parent
            return json.loads(raw.decode())
        except Exception:
            return None

    def _dispatch_to_worker(self, model_id: str) -> None:
        """Swap self.response_generator to the per-model worker and call parent do_POST."""
        resolved = _resolve_model_path(model_id)
        worker = _get_or_create_worker(resolved)
        # Rewrite the model field in the request body so the parent handler
        # passes the resolved HF repo ID (not the Ollama-style name) to
        # model_provider.load() inside the worker thread.
        content_length = int(self.headers.get("Content-Length", 0))
        saved_rfile = self.rfile
        raw = self.rfile.read(content_length)
        try:
            body = json.loads(raw)
            body["model"] = resolved
            rewritten = json.dumps(body).encode()
            self.rfile = io.BytesIO(rewritten)
            self.headers.replace_header("Content-Length", str(len(rewritten)))
        except Exception:
            # Restore original bytes on any failure
            self.rfile = io.BytesIO(raw)
        original_rg = self.response_generator
        self.response_generator = worker
        try:
            super().do_POST()
        finally:
            self.response_generator = original_rg

    # thinking_budget / enable_thinking — Option B implementation:
    # Route through invoke.invoke() directly when these params are present.
    # This bypasses mlx_lm's ResponseGenerator, losing streaming and tool-calling,
    # which is acceptable for deep reasoning calls that don't stream.
    # Option A (intercepting the chat-template call) would require patching mlx_lm internals.

    def _handle_thinking_completion(self, body: dict):
        """Handle /v1/chat/completions when thinking_budget or enable_thinking is present.

        Routes through invoke.invoke() directly (non-streaming, no tool-calling).
        Returns an OpenAI-compatible response.
        """
        from invoke import invoke as mlx_invoke

        model_path = body.get("model", self._default_model)
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 4096)
        thinking_budget = body.get("thinking_budget", 0)
        enable_thinking = body.get("enable_thinking", True)
        chat_template_kwargs = body.get("chat_template_kwargs") or {}

        try:
            text, tok_in, tok_out = mlx_invoke(
                model_path,
                messages,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                enable_thinking=enable_thinking,
                chat_template_kwargs=chat_template_kwargs,
            )
        except Exception as e:
            log.error("invoke failed: %s", e)
            self._json_response(500, {"error": str(e)})
            return

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_path,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": tok_in,
                "completion_tokens": tok_out,
                "total_tokens": tok_in + tok_out,
            },
        }
        self._json_response(200, response)

    # ── Ollama route handlers ────────────────────────────────────────────────

    @staticmethod
    def _ollama_model_details(name: str) -> dict:
        """Derive Ollama-style model details from a HuggingFace repo name.

        Heuristically extracts family, parameter_size, and quantization_level
        from names like 'mlx-community/Qwen3.5-9B-MLX-4bit' or 'gemma-4-e4b-it-4bit'.
        """
        base = name.split("/")[-1].lower()
        # Parameter size: look for patterns like 9b, 27b, 0.6b, 3b, a3b (MoE active)
        param_match = re.search(r'(\d+(?:\.\d+)?b(?:-a\d+b)?)', base)
        parameter_size = param_match.group(1).upper() if param_match else ""
        # Quantization: look for 4bit, 8bit, 6bit, 3bit, bf16, fp16, mxfp4, mxfp8
        quant_match = re.search(r'(\d+bit|bf16|fp16|mxfp\d+|nvfp\d+|optiq-\d+bit)', base)
        quantization_level = quant_match.group(1) if quant_match else ""
        # Family: first meaningful word before the size token
        family_match = re.match(r'([a-z][a-z0-9._-]+?)[\s\-_]?\d', base)
        family = family_match.group(1).rstrip("-_") if family_match else base.split("-")[0]
        return {
            "parent_model": "",
            "format": "mlx",
            "family": family,
            "families": [family],
            "parameter_size": parameter_size,
            "quantization_level": quantization_level,
        }

    def _handle_ollama_ps(self):
        """Handle GET /api/ps — list currently loaded (resident) models."""
        entries = []
        for stat in _model_cache.stats():
            name = stat["id"]
            entries.append({
                "name": name,
                "model": name,
                "size": stat["size_gb"] and int(stat["size_gb"] * 1e9),
                "size_vram": int(stat["size_gb"] * 1e9),  # unified memory on Apple Silicon
                "digest": "",
                "details": self._ollama_model_details(name),
                "expires_at": "",  # we don't expire resident models automatically
            })
        self._json_response(200, {"models": entries})

    def _handle_ollama_show(self):
        """Handle POST /api/show — return metadata for a named model."""
        try:
            raw = self._read_body()
            body = json.loads(raw.decode())
        except Exception as e:
            self._json_response(400, {"error": f"Invalid JSON: {e}"})
            return

        name = _resolve_model_path(body.get("model") or self._default_model or "")
        if not name:
            self._json_response(400, {"error": "model is required"})
            return

        known = set(_loaded_model_names)
        if self._default_model:
            known.add(self._default_model)
        if name not in known:
            self._json_response(404, {"error": f"model '{name}' not found"})
            return

        details = self._ollama_model_details(name)
        self._json_response(200, {
            "modelfile": f"# MLX model: {name}\nFROM {name}\n",
            "parameters": "",
            "template": "",
            "details": details,
            "model_info": {},
        })

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
            self.headers.replace_header("Content-Length", str(len(remapped)))
            self.rfile = io.BytesIO(remapped)
            self._handle_embeddings()
        except Exception as e:
            log.error("Ollama embeddings failed: %s", e)
            self._json_response(500, {"error": str(e)})

    def _handle_ollama_embed(self):
        """Handle POST /api/embed — newer Ollama batched embed endpoint.

        Accepts {"model": str, "input": str | list[str]} and returns
        {"model": str, "embeddings": [[float, ...], ...]} as Ollama specifies.
        """
        try:
            raw = self._read_body()
            body = json.loads(raw.decode())

            model_path = body.get("model", "")
            input_value = body.get("input", "")

            if not model_path:
                self._json_response(400, {"error": "model is required"})
                return

            texts = input_value if isinstance(input_value, list) else [input_value]
            if not texts or not any(texts):
                self._json_response(400, {"error": "input is required"})
                return

            model, tokenizer = _get_embed_model(model_path)

            embeddings = []
            with _embed_lock:
                for text in texts:
                    input_ids = tokenizer.encode(text, return_tensors="mlx")
                    outputs = model(input_ids)
                    embeddings.append(outputs.text_embeds[0].tolist())

            self._json_response(200, {"model": model_path, "embeddings": embeddings})
        except Exception as e:
            log.error("Ollama embed failed: %s", e)
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

        model = _resolve_model_path(ollama_body.get("model") or self._default_model)
        log.info("  model=%s", model)
        openai_body = OllamaAdapter.translate_request(ollama_body, wrap_prompt=wrap_prompt)
        # Ensure the resolved model path is used downstream
        if model:
            openai_body["model"] = model

        # Propagate thinking params from top-level or nested options{}
        for key in ("thinking_budget", "enable_thinking"):
            if key in ollama_body:
                openai_body[key] = ollama_body[key]
            elif key in ollama_body.get("options", {}):
                openai_body[key] = ollama_body["options"][key]
        if "chat_template_kwargs" in ollama_body:
            openai_body["chat_template_kwargs"] = ollama_body["chat_template_kwargs"]

        # /no_think prefix in the system message → disable thinking via chat template
        # (Qwen3 also honors this as a prompt token; we additionally set enable_thinking=False
        # so the tokenizer's apply_chat_template produces the right template variant.)
        for msg in openai_body.get("messages", []):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and content.lstrip().startswith("/no_think"):
                    openai_body.setdefault("enable_thinking", False)
                break

        # Default thinking off for Qwen3/3.5 models unless explicitly enabled.
        # These models emit <think>...</think> blocks by default which breaks
        # structured-output consumers (e.g. LightRAG entity extraction).
        # Use chat_template_kwargs to disable thinking at the template level
        # (the /no_think prompt prefix is unreliable with some Qwen3.5 variants).
        _qwen3_thinking_injected = False
        if "enable_thinking" not in openai_body and "thinking_budget" not in openai_body \
                and "chat_template_kwargs" not in openai_body:
            model_lower = (model or "").lower()
            if "qwen3" in model_lower:
                openai_body["chat_template_kwargs"] = {"enable_thinking": False}
                _qwen3_thinking_injected = True

        # Route through invoke() when any thinking/template-control field is present,
        # UNLESS we auto-injected it for Qwen3 (which should use the normal
        # streaming path to preserve Ollama format compatibility).
        if not _qwen3_thinking_injected and (
            "thinking_budget" in openai_body
            or "enable_thinking" in openai_body
            or "chat_template_kwargs" in openai_body
        ):
            self._handle_thinking_completion(openai_body)
            return

        is_streaming = openai_body.get("stream", True)

        # Prepare to call parent's do_POST with remapped path + body
        encoded_body = json.dumps(openai_body).encode()
        original_path = self.path
        original_rfile = self.rfile
        original_wfile = self.wfile

        self.path = "/v1/chat/completions"
        self.headers.replace_header("Content-Length", str(len(encoded_body)))
        self.rfile = io.BytesIO(encoded_body)

        try:
            if is_streaming:
                # Intercept wfile to translate SSE → NDJSON
                self.wfile = _StreamTranslatorWfile(original_wfile, model)
                if _server_args is not None and model:
                    self._dispatch_to_worker(model)
                else:
                    super().do_POST()
                # Flush any remaining buffered bytes
                self.wfile.flush()
            else:
                # Buffer the entire OpenAI response, then translate
                buf = io.BytesIO()
                self.wfile = buf
                if _server_args is not None and model:
                    self._dispatch_to_worker(model)
                else:
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


def _list_local_models() -> None:
    """Print HuggingFace models cached locally and exit."""
    hf_cache = os.path.expanduser(
        os.environ.get("HF_HOME", os.environ.get("HUGGINGFACE_HUB_CACHE", "~/.cache/huggingface/hub"))
    )
    if not os.path.isdir(hf_cache):
        print(f"No HuggingFace cache found at {hf_cache}")
        return
    models = []
    for entry in sorted(os.listdir(hf_cache)):
        if entry.startswith("models--"):
            # models--org--repo  →  org/repo
            parts = entry[len("models--"):].split("--", 1)
            if len(parts) == 2:
                models.append(f"{parts[0]}/{parts[1]}")
            else:
                models.append(parts[0])
    if not models:
        print("No models found in local HuggingFace cache.")
        return
    print(f"Local models ({len(models)}) in {hf_cache}:")
    for m in models:
        print(f"  {m}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX inference server with embeddings and LoRA")
    parser.add_argument("--model", type=str, default=None,
                        help="Default model to preload (HuggingFace path)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=11434,
                        help="Port (default: 11434 — matches Ollama)")
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
    # Phase 2.1: multi-model preload
    parser.add_argument("--preload", action="append", metavar="MODEL",
                        help="Chat model to preload at startup (can be specified multiple times)")
    parser.add_argument("--preload-embedding", action="append", metavar="MODEL",
                        help="Embedding model to preload at startup (can be specified multiple times)")
    _default_config = os.path.expanduser("~/.config/ai-mlx-server/models.yaml")
    parser.add_argument("--models-config", type=str,
                        default=_default_config if os.path.exists(_default_config) else None,
                        metavar="PATH",
                        help="Path to YAML config file listing models to preload "
                             "(default: ~/.config/ai-mlx-server/models.yaml if it exists)")
    # Phase 2.2: LRU eviction limits
    parser.add_argument("--max-resident-models", type=int, default=None, metavar="N",
                        help="Maximum number of models to keep in memory")
    parser.add_argument("--max-resident-gb", type=float, default=None, metavar="N",
                        help="Maximum total model memory in GB before eviction")
    # Phase 5.1: bearer-token authentication
    parser.add_argument("--api-key", type=str, default=None, metavar="KEY",
                        help="API key for bearer-token auth (fallback: MLX_API_KEY env var)")
    parser.add_argument("--auth-health", action="store_true",
                        help="Require auth for /health (default: /health is public)")
    parser.add_argument("--auth-metrics", action="store_true",
                        help="Require auth for /metrics (default: /metrics is public)")
    parser.add_argument("--allowed-origins", nargs="*", default=[],
                        help="CORS allowed origins (passed through to mlx_lm)")
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow automatic model downloads from HuggingFace (default: offline-only)")
    parser.add_argument("--list", action="store_true",
                        help="List locally cached HuggingFace models and exit")

    args = parser.parse_args()

    if args.list:
        _list_local_models()
        return
    global _server_args
    _server_args = args
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Prevent accidental HF downloads unless --allow-download is set.
    # The embedding loader handles its own offline-first + fallback logic.
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Configure the unified model cache with eviction limits
    global _model_cache
    max_bytes = int(args.max_resident_gb * 1e9) if args.max_resident_gb is not None else None
    _model_cache = ModelCache(
        max_models=args.max_resident_models,
        max_bytes=max_bytes,
        on_evict=_tear_down_worker,
    )

    model_provider = ModelProvider(args)
    prompt_cache = LRUPromptCache(args.prompt_cache_size or 1)
    response_generator = ResponseGenerator(model_provider, prompt_cache)

    # Inject the default model name and response generator into the handler class.
    MLXAPIHandler._default_model = args.model or ""
    MLXAPIHandler._response_generator = response_generator

    # Phase 5.1: inject auth config
    api_key = args.api_key or os.environ.get("MLX_API_KEY", "")
    MLXAPIHandler._api_key = api_key
    MLXAPIHandler._auth_health = args.auth_health
    MLXAPIHandler._auth_metrics = args.auth_metrics
    if api_key:
        log.info("API key authentication enabled")

    # Phase 2.1: collect all models to preload from CLI args and config file
    preload_chat: list[str] = list(args.preload or [])
    preload_embed: list[str] = list(args.preload_embedding or [])

    if args.models_config:
        try:
            config_models, config_aliases = _parse_models_config(args.models_config)
            if config_aliases:
                _model_aliases.update(config_aliases)
                log.info("Loaded %d model alias(es): %s", len(config_aliases),
                         ", ".join(f"{k} → {v}" for k, v in config_aliases.items()))
            for entry in config_models:
                model_id = entry.get("id", "")
                role = entry.get("role", "chat")
                if not model_id:
                    continue
                if role == "embedding":
                    pass  # embedding models are not preloaded by default; use --preload-embedding
                else:
                    preload_chat.append(model_id)
        except Exception as e:
            log.error("Failed to parse models config %s: %s", args.models_config, e)

    # Preload chat models sequentially
    for model_id in preload_chat:
        log.info("Preloading chat model: %s", model_id)
        t0 = time.time()
        try:
            # Force model load via ModelProvider by temporarily swapping args.model
            orig_model = args.model
            args.model = model_id
            model_provider_tmp = ModelProvider(args)
            model_provider_tmp.load()
            args.model = orig_model
            est = estimate_model_bytes(model_id)
            _model_cache.put(model_id, model_provider_tmp, est_bytes=est, pinned=True, role="chat")
            _loaded_model_names.add(model_id)
            log.info("Preloaded chat model %s in %.1fs", model_id, time.time() - t0)
            # Create the per-model worker eagerly so requests route immediately.
            _get_or_create_worker(model_id)
            log.info("Worker ready for %s", model_id)
        except Exception as e:
            log.error("Failed to preload chat model %s: %s", model_id, e)

    # Preload embedding models sequentially
    for model_id in preload_embed:
        log.info("Preloading embedding model: %s", model_id)
        t0 = time.time()
        try:
            _get_embed_model(model_id)
            _model_cache.pin(model_id)
            log.info("Preloaded embedding model %s in %.1fs", model_id, time.time() - t0)
        except Exception as e:
            log.error("Failed to preload embedding model %s: %s", model_id, e)

    handler = partial(
        MLXAPIHandler,
        response_generator,
        system_fingerprint="mlx-server",
    )

    server = DualStackHTTPServer((args.host, args.port), handler)
    log.info("MLX server listening on %s:%d", args.host, args.port)
    if args.model:
        log.info("Default model: %s", args.model)
    if args.adapter_path:
        log.info("LoRA adapter: %s", args.adapter_path)
    log.info(
        "Endpoints: /v1/chat/completions, /v1/embeddings, /health, "
        "/api/generate, /api/chat, /api/embeddings, /api/embed, /api/tags, /api/version"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down…")
        # Tear down all per-model workers first.
        with _workers_lock:
            worker_snapshot = dict(_model_workers)
        for mid, w in worker_snapshot.items():
            log.info("Stopping worker for %s", mid)
            w.stop_and_join()
        with _workers_lock:
            _model_workers.clear()
        response_generator.stop_and_join()
        server.server_close()


if __name__ == "__main__":
    main()
