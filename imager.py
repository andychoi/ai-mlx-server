#!/usr/bin/env python3
"""MLX image-generation server — OpenAI-compatible `/v1/images/generations` and `/v1/images/edits`.

Runs on Apple Silicon. Sibling to `server.py` (which serves chat/embeddings).
Backed by mflux (https://github.com/filipstrand/mflux). Default txt2img model
is Qwen-Image-2512; default edit model is Qwen-Image-Edit-2511.

Usage:
    python imager.py
    python imager.py --model mlx-community/Qwen-Image-2512-4bit
    python imager.py --list   # show locally cached HuggingFace models and exit
"""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import logging
import os
import socketserver
import tempfile
import threading
import time
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler

# Reuse helpers from server.py — same repo, same conventions.
from server import (
    DualStackHTTPServer,
    _resolve_model_path,
    _local_hf_models,
    _list_local_models,
    _hf_id_to_ollama_name,
    _hf_cache_dir,
    _parse_models_config,
    _model_aliases,
)
from cache import ModelCache, estimate_model_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mlx-imager")

_SERVER_START_TIME = time.time()
_DEFAULT_MODEL = "mlx-community/Qwen-Image-2512-4bit"
# Qwen-Image-Edit has no MLX-quantized port on HuggingFace yet (as of May 2026),
# so the edit endpoint defaults to upstream BF16 and lets mflux quantize at load time.
_DEFAULT_EDIT_MODEL = "Qwen/Qwen-Image-Edit-2511"

# Version read from pyproject.toml at import time.
_SERVER_VERSION = "0.1.0"
try:
    import tomllib
    _toml_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    with open(_toml_path, "rb") as _f:
        _SERVER_VERSION = tomllib.load(_f)["project"]["version"]
except Exception:
    pass

# Optional Prometheus metrics.
try:
    from metrics import (
        requests_total,
        request_duration_seconds,
        resident_models,
        resident_bytes,
        queue_depth as _queue_depth_gauge,
        images_generated_total,
        generation_duration_seconds,
        inference_steps_total,
        image_failures_total,
    )
    _metrics_enabled = True
except ImportError:
    _metrics_enabled = False


# ── Model cache + concurrency primitives ────────────────────────────────────
_model_cache = ModelCache()
_load_locks: dict[str, threading.Lock] = {}
_load_locks_lock = threading.Lock()
# Single inference lock — MLX uses one GPU; serialize generations across models.
_gen_lock = threading.Lock()
_queue_depth = 0
_queue_depth_lock = threading.Lock()
_server_args: argparse.Namespace | None = None


# ── mflux loaders (verified against mflux 0.17.5) ───────────────────────────
def _load_image_model(model_path: str, quantize: int):
    """Load an mflux Qwen-Image model for text-to-image.

    Verified signature for mflux 0.17.x:
        QwenImage(quantize=int|None, model_path=str|None, lora_paths=None, lora_scales=None)
    """
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
    return QwenImage(quantize=quantize, model_path=model_path)


def _load_edit_model(model_path: str, quantize: int):
    """Load an mflux Qwen-Image-Edit model for image editing.

    Verified signature for mflux 0.17.x:
        QwenImageEdit(quantize=int|None, model_path=str|None, lora_paths=None, lora_scales=None)
    """
    from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
    return QwenImageEdit(quantize=quantize, model_path=model_path)


def _get_cached_model(model_path: str, *, role: str, loader):
    """Return a cached mflux model for the resolved path, loading on first miss.

    `role` distinguishes txt2img ("image") and edit ("image-edit") in ModelCache stats;
    `loader(resolved_path, quantize)` builds the model when not cached.
    """
    resolved = _resolve_model_path(model_path)
    cached = _model_cache.get(resolved)
    if cached is not None:
        return cached, resolved

    with _load_locks_lock:
        lk = _load_locks.setdefault(resolved, threading.Lock())
    with lk:
        cached = _model_cache.get(resolved)
        if cached is not None:
            return cached, resolved
        t0 = time.time()
        log.info("Loading %s model %s …", role, resolved)
        quantize = int(getattr(_server_args, "quantize", 4)) if _server_args else 4
        model = loader(resolved, quantize=quantize)
        est = estimate_model_bytes(resolved)
        _model_cache.put(resolved, model, est_bytes=est, role=role)
        log.info("Loaded %s in %.1fs (~%.1f GB)", resolved, time.time() - t0, est / 1e9)
        return model, resolved


def _get_image_model(model_path: str):
    return _get_cached_model(model_path, role="image", loader=_load_image_model)


def _get_edit_model(model_path: str):
    return _get_cached_model(model_path, role="image-edit", loader=_load_edit_model)


def _generate_image(model, prompt: str, *, negative_prompt: str, width: int, height: int,
                    steps: int, guidance: float, seed: int):
    """Call mflux's QwenImage.generate_image. Verified kwargs for mflux 0.17.x."""
    return model.generate_image(
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
        guidance=guidance,
    )


def _generate_edit(model, prompt: str, *, negative_prompt: str, image_paths: list[str],
                   width: int | None, height: int | None, steps: int, guidance: float, seed: int):
    """Call mflux's QwenImageEdit.generate_image with init image(s).

    Verified signature for mflux 0.17.x:
        QwenImageEdit.generate_image(seed, prompt, image_paths: list[str],
            num_inference_steps=4, height, width, guidance=4.0,
            scheduler='linear', negative_prompt=None, ...)
    """
    return model.generate_image(
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_paths=image_paths,
        num_inference_steps=steps,
        width=width,
        height=height,
        guidance=guidance,
    )


def _extract_pil_image(result):
    """mflux returns a GeneratedImage with a .image attribute holding the PIL.Image."""
    image = getattr(result, "image", None)
    if image is None:
        # Defensive: a future mflux release could return the PIL.Image directly.
        if hasattr(result, "save"):
            return result
        raise RuntimeError(f"mflux returned unrecognized image object: {type(result)!r}")
    return image


# ── Request validation ─────────────────────────────────────────────────────
class _BadRequest(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.message = message


def _parse_size(size: str | None, default_w: int, default_h: int, max_pixels: int) -> tuple[int, int]:
    if size is None or size == "auto":
        return default_w, default_h
    if not isinstance(size, str) or "x" not in size:
        raise _BadRequest(f"size must be WIDTHxHEIGHT (got {size!r})")
    try:
        w_s, h_s = size.split("x", 1)
        w, h = int(w_s), int(h_s)
    except ValueError:
        raise _BadRequest(f"size must be WIDTHxHEIGHT (got {size!r})")
    if not (64 <= w <= 2048 and 64 <= h <= 2048):
        raise _BadRequest(f"size dimensions must be in [64, 2048] (got {w}x{h})")
    if w % 8 != 0 or h % 8 != 0:
        raise _BadRequest(f"size dimensions must be multiples of 8 (got {w}x{h})")
    if w * h > max_pixels:
        raise _BadRequest(f"size {w}x{h} exceeds max_pixels={max_pixels}")
    return w, h


def _validate_request(body: dict, args: argparse.Namespace) -> dict:
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _BadRequest("`prompt` is required and must be a non-empty string")

    n = body.get("n", 1)
    if not isinstance(n, int) or n < 1:
        raise _BadRequest("`n` must be a positive integer")
    if n > args.max_batch_n:
        raise _BadRequest(f"`n` must be <= {args.max_batch_n}")

    response_format = body.get("response_format", "b64_json")
    if response_format == "url":
        raise _BadRequest(
            "response_format='url' is not supported by this server (no blob store). "
            "Use 'b64_json'."
        )
    if response_format != "b64_json":
        raise _BadRequest(f"response_format must be 'b64_json' (got {response_format!r})")

    width, height = _parse_size(body.get("size"), args.width, args.height, args.max_pixels)

    steps = body.get("steps", args.steps)
    if not isinstance(steps, int) or steps < 1 or steps > args.max_steps:
        raise _BadRequest(f"`steps` must be int in [1, {args.max_steps}]")

    guidance = body.get("guidance_scale", args.guidance)
    if not isinstance(guidance, (int, float)) or not (0 <= guidance <= 20):
        raise _BadRequest("`guidance_scale` must be a number in [0, 20]")

    seed = body.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise _BadRequest("`seed` must be an integer")

    negative_prompt = body.get("negative_prompt", "") or ""
    if not isinstance(negative_prompt, str):
        raise _BadRequest("`negative_prompt` must be a string")

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "n": n,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": float(guidance),
        "seed": seed,
        "model": body.get("model") or args.model,
    }


def _parse_multipart(content_type: str, body: bytes) -> dict:
    """Parse multipart/form-data into {field_name: str_or_bytes}.

    Uses stdlib email parser since `cgi` was removed in Python 3.13.
    For text fields, returns str; for file fields, returns bytes.
    """
    if not content_type or "multipart/form-data" not in content_type.lower():
        raise _BadRequest("Content-Type must be multipart/form-data for /v1/images/edits")
    preamble = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    msg = BytesParser(policy=policy.default).parsebytes(preamble + body)
    if not msg.is_multipart():
        raise _BadRequest("Failed to parse multipart body")
    fields: dict = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_param("filename", header="content-disposition")
        if filename:
            fields[name] = part.get_payload(decode=True)
        else:
            payload = part.get_payload(decode=True)
            fields[name] = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
    return fields


def _coerce_int(form: dict, key: str, default):
    if key not in form:
        return default
    try:
        return int(form[key])
    except (TypeError, ValueError):
        raise _BadRequest(f"`{key}` must be an integer")


def _coerce_float(form: dict, key: str, default):
    if key not in form:
        return default
    try:
        return float(form[key])
    except (TypeError, ValueError):
        raise _BadRequest(f"`{key}` must be a number")


def _validate_edit_request(form: dict, args: argparse.Namespace) -> dict:
    """Validate fields from a parsed multipart form for /v1/images/edits."""
    prompt = form.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _BadRequest("`prompt` is required and must be a non-empty string")

    image_bytes = form.get("image")
    if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) == 0:
        raise _BadRequest("`image` file part is required")

    n = _coerce_int(form, "n", 1)
    if n < 1 or n > args.max_batch_n:
        raise _BadRequest(f"`n` must be int in [1, {args.max_batch_n}]")

    response_format = form.get("response_format", "b64_json")
    if response_format == "url":
        raise _BadRequest(
            "response_format='url' is not supported by this server (no blob store). Use 'b64_json'."
        )
    if response_format != "b64_json":
        raise _BadRequest(f"response_format must be 'b64_json' (got {response_format!r})")

    # Size is optional for edit: omitting it lets mflux use the source image's dimensions.
    size = form.get("size")
    if size in (None, "", "auto"):
        width = height = None
    else:
        width, height = _parse_size(size, args.width, args.height, args.max_pixels)

    steps = _coerce_int(form, "steps", args.steps)
    if steps < 1 or steps > args.max_steps:
        raise _BadRequest(f"`steps` must be int in [1, {args.max_steps}]")

    guidance = _coerce_float(form, "guidance_scale", args.guidance)
    if not (0 <= guidance <= 20):
        raise _BadRequest("`guidance_scale` must be a number in [0, 20]")

    seed_val = form.get("seed")
    seed = _coerce_int(form, "seed", None) if seed_val is not None else None

    negative_prompt = form.get("negative_prompt", "") or ""
    if not isinstance(negative_prompt, str):
        raise _BadRequest("`negative_prompt` must be a string")

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": bytes(image_bytes),
        "n": n,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": float(guidance),
        "seed": seed,
        "model": form.get("model") or args.edit_model,
    }


def _encode_image(pil_image, fmt: str, jpeg_quality: int) -> str:
    buf = io.BytesIO()
    save_kwargs: dict = {}
    f = fmt.upper()
    if f == "JPEG":
        save_kwargs["quality"] = jpeg_quality
    elif f == "PNG":
        save_kwargs["optimize"] = True
    pil_image.save(buf, format=f, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── HTTP handler ────────────────────────────────────────────────────────────
class ImagerAPIHandler(BaseHTTPRequestHandler):
    # Injected in main()
    _api_key: str = ""
    _auth_health: bool = False
    _auth_metrics: bool = False
    _args: argparse.Namespace | None = None

    server_version = f"ai-mlx-imager/{_SERVER_VERSION}"

    # CORS / auth ----------------------------------------------------------
    def _check_auth(self, path: str) -> bool:
        """Bearer-token gate. Copy of server.MLXAPIHandler._check_auth."""
        if not self._api_key:
            return True
        if path == "/health" and not self._auth_health:
            return True
        if path == "/metrics" and not self._auth_metrics:
            return True
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        provided_key = auth_header[len("Bearer "):]
        return hmac.compare_digest(provided_key, self._api_key)

    def _set_cors_headers(self):
        allowed = (self._args.allowed_origins if self._args else []) or []
        origin = self.headers.get("Origin", "")
        if "*" in allowed or origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin or "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def _json(self, status: int, body: dict):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    # Routing --------------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if not self._check_auth(self.path):
            self._json(401, {"error": "Unauthorized"}); return
        if self.path == "/health":
            self._handle_health(); return
        if self.path == "/metrics":
            self._handle_metrics(); return
        if self.path in ("/api/version", "/v1/version"):
            self._json(200, {"version": _SERVER_VERSION}); return
        if self.path == "/api/tags":
            self._handle_tags(); return
        self.send_error(404)

    def do_POST(self):
        if not self._check_auth(self.path):
            self._json(401, {"error": "Unauthorized"}); return
        if self.path == "/v1/images/generations":
            self._handle_generations(); return
        if self.path == "/v1/images/edits":
            self._handle_edits(); return
        self.send_error(404)

    # Handlers -------------------------------------------------------------
    def _handle_health(self):
        import psutil
        mem = psutil.virtual_memory()
        body = {
            "status": "ok",
            "version": _SERVER_VERSION,
            "uptime_seconds": round(time.time() - _SERVER_START_TIME, 1),
            "models": _model_cache.stats(),
            "memory": {
                "total_gb": round(mem.total / 1e9, 2),
                "available_gb": round(mem.available / 1e9, 2),
                "used_pct": mem.percent,
            },
            "queue_depth": _queue_depth,
        }
        self._json(200, body)

    def _handle_metrics(self):
        if not _metrics_enabled:
            self._json(503, {"error": "prometheus_client not installed"}); return
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        resident_models.set(len(_model_cache))
        resident_bytes.set(_model_cache.total_bytes())
        _queue_depth_gauge.set(_queue_depth)
        payload = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_tags(self):
        """Ollama-style /api/tags — list cached image-capable models.

        Filters by name for now; mflux supports several diffusion model families.
        Not advertised in README; included for parity with server.py.
        """
        models = []
        for hf_id in _local_hf_models():
            low = hf_id.lower()
            if "qwen-image" in low or "flux" in low or "z-image" in low or "fibo" in low:
                models.append({
                    "name": _hf_id_to_ollama_name(hf_id),
                    "model": hf_id,
                    "modified_at": datetime.now(timezone.utc).isoformat(),
                })
        self._json(200, {"models": models})

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise _BadRequest("Empty request body")
        try:
            raw = self.rfile.read(length)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise _BadRequest(f"Invalid JSON: {e}")

    def _handle_generations(self):
        global _queue_depth
        route = "/v1/images/generations"
        t_total = time.time()
        model_label = "?"
        try:
            body = self._read_body()
            params = _validate_request(body, self._args)
            model_label = params["model"]

            with _queue_depth_lock:
                _queue_depth += 1
            try:
                with _gen_lock:
                    model, resolved = _get_image_model(params["model"])
                    images_b64: list[str] = []
                    base_seed = params["seed"] if params["seed"] is not None else int(time.time() * 1000) & 0xFFFFFFFF
                    for i in range(params["n"]):
                        t_gen = time.time()
                        result = _generate_image(
                            model,
                            prompt=params["prompt"],
                            negative_prompt=params["negative_prompt"],
                            width=params["width"],
                            height=params["height"],
                            steps=params["steps"],
                            guidance=params["guidance"],
                            seed=base_seed + i,
                        )
                        pil_image = _extract_pil_image(result)
                        b64 = _encode_image(pil_image, self._args.output_format, self._args.jpeg_quality)
                        images_b64.append(b64)
                        if _metrics_enabled:
                            generation_duration_seconds.labels(model=resolved).observe(time.time() - t_gen)
                            inference_steps_total.labels(model=resolved).inc(params["steps"])
                            images_generated_total.labels(model=resolved).inc()
            finally:
                with _queue_depth_lock:
                    _queue_depth -= 1

            response = {
                "created": int(time.time()),
                "data": [{"b64_json": b, "revised_prompt": None} for b in images_b64],
            }
            self._json(200, response)
            if _metrics_enabled:
                requests_total.labels(route=route, model=model_label, status="200").inc()
                request_duration_seconds.labels(route=route, model=model_label).observe(time.time() - t_total)

        except _BadRequest as e:
            log.info("400 %s: %s", route, e.message)
            self._json(e.status, {"error": e.message})
            if _metrics_enabled:
                requests_total.labels(route=route, model=model_label, status=str(e.status)).inc()
                image_failures_total.labels(model=model_label, reason="bad_request").inc()
        except Exception as e:
            log.exception("Image generation failed: %s", e)
            self._json(500, {"error": str(e)})
            if _metrics_enabled:
                requests_total.labels(route=route, model=model_label, status="500").inc()
                image_failures_total.labels(model=model_label, reason="internal").inc()


    def _read_raw_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise _BadRequest("Empty request body")
        return self.rfile.read(length)

    def _handle_edits(self):
        global _queue_depth
        route = "/v1/images/edits"
        t_total = time.time()
        model_label = "?"
        tmp_path: str | None = None
        try:
            raw = self._read_raw_body()
            form = _parse_multipart(self.headers.get("Content-Type", ""), raw)
            params = _validate_edit_request(form, self._args)
            model_label = params["model"]

            # mflux's edit loader takes file paths, so persist the upload briefly.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tf.write(params["image"])
                tmp_path = tf.name

            with _queue_depth_lock:
                _queue_depth += 1
            try:
                with _gen_lock:
                    model, resolved = _get_edit_model(params["model"])
                    images_b64: list[str] = []
                    base_seed = params["seed"] if params["seed"] is not None else int(time.time() * 1000) & 0xFFFFFFFF
                    for i in range(params["n"]):
                        t_gen = time.time()
                        result = _generate_edit(
                            model,
                            prompt=params["prompt"],
                            negative_prompt=params["negative_prompt"],
                            image_paths=[tmp_path],
                            width=params["width"],
                            height=params["height"],
                            steps=params["steps"],
                            guidance=params["guidance"],
                            seed=base_seed + i,
                        )
                        pil_image = _extract_pil_image(result)
                        b64 = _encode_image(pil_image, self._args.output_format, self._args.jpeg_quality)
                        images_b64.append(b64)
                        if _metrics_enabled:
                            generation_duration_seconds.labels(model=resolved).observe(time.time() - t_gen)
                            inference_steps_total.labels(model=resolved).inc(params["steps"])
                            images_generated_total.labels(model=resolved).inc()
            finally:
                with _queue_depth_lock:
                    _queue_depth -= 1

            response = {
                "created": int(time.time()),
                "data": [{"b64_json": b, "revised_prompt": None} for b in images_b64],
            }
            self._json(200, response)
            if _metrics_enabled:
                requests_total.labels(route=route, model=model_label, status="200").inc()
                request_duration_seconds.labels(route=route, model=model_label).observe(time.time() - t_total)

        except _BadRequest as e:
            log.info("400 %s: %s", route, e.message)
            self._json(e.status, {"error": e.message})
            if _metrics_enabled:
                requests_total.labels(route=route, model=model_label, status=str(e.status)).inc()
                image_failures_total.labels(model=model_label, reason="bad_request").inc()
        except Exception as e:
            log.exception("Image edit failed: %s", e)
            self._json(500, {"error": str(e)})
            if _metrics_enabled:
                requests_total.labels(route=route, model=model_label, status="500").inc()
                image_failures_total.labels(model=model_label, reason="internal").inc()
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


class ThreadedDualStackHTTPServer(socketserver.ThreadingMixIn, DualStackHTTPServer):
    """Threaded HTTP server so /health and /metrics stay responsive during inference."""
    daemon_threads = True
    allow_reuse_address = True


# ── CLI ────────────────────────────────────────────────────────────────────
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLX image-generation server (OpenAI-compatible)")

    # Reused-from-server.py flags ----------------------------------------
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                   help=f"Default txt2img model (HF repo ID or Ollama-style name). Default: {_DEFAULT_MODEL}")
    p.add_argument("--edit-model", type=str, default=_DEFAULT_EDIT_MODEL,
                   help=f"Default edit model for /v1/images/edits. Default: {_DEFAULT_EDIT_MODEL} "
                        f"(no MLX-quantized port exists yet; mflux quantizes BF16 weights on load).")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=11435, help="Port (default: 11435 — LLM server is 11434)")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--api-key", type=str, default=None, metavar="KEY",
                   help="Bearer-token API key (fallback: MLX_IMAGER_API_KEY env var)")
    p.add_argument("--auth-health", action="store_true",
                   help="Require auth for /health (default: /health is public)")
    p.add_argument("--auth-metrics", action="store_true",
                   help="Require auth for /metrics (default: /metrics is public)")
    p.add_argument("--allowed-origins", nargs="*", default=[], help="CORS allowed origins")
    p.add_argument("--allow-download", action="store_true",
                   help="Allow HuggingFace downloads (default: offline-only)")
    p.add_argument("--list", action="store_true",
                   help="List locally cached HuggingFace models and exit")
    p.add_argument("--preload", action="append", metavar="MODEL",
                   help="Image model to preload at startup (can be repeated)")
    _default_config = os.path.expanduser("~/.config/mlx-server/models.yaml")
    p.add_argument("--models-config", type=str,
                   default=_default_config if os.path.exists(_default_config) else None,
                   metavar="PATH",
                   help="YAML config file (re-uses server.py format; only role=image entries are read)")
    p.add_argument("--max-resident-models", type=int, default=None, metavar="N")
    p.add_argument("--max-resident-gb", type=float, default=None, metavar="N")

    # Image-specific flags -----------------------------------------------
    p.add_argument("--quantize", type=int, default=4, choices=[3, 4, 6, 8],
                   help="Weight quantization bits (default: 4)")
    p.add_argument("--steps", type=int, default=20, help="Default inference steps")
    p.add_argument("--max-steps", type=int, default=60, help="Per-request steps ceiling")
    p.add_argument("--guidance", type=float, default=4.0, help="Default guidance scale")
    p.add_argument("--width", type=int, default=1024, help="Default image width")
    p.add_argument("--height", type=int, default=1024, help="Default image height")
    p.add_argument("--max-pixels", type=int, default=2_097_152,
                   help="Maximum width*height per image (default: 2_097_152 = 1448²)")
    p.add_argument("--max-batch-n", type=int, default=4, help="Maximum n per request")
    p.add_argument("--output-format", type=str, default="png", choices=["png", "jpeg", "webp"])
    p.add_argument("--jpeg-quality", type=int, default=92)
    return p


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    if args.list:
        _list_local_models()
        return

    global _server_args, _model_cache
    _server_args = args
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.model:
        args.model = _resolve_model_path(args.model)
    if args.edit_model:
        args.edit_model = _resolve_model_path(args.edit_model)

    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    max_bytes = int(args.max_resident_gb * 1e9) if args.max_resident_gb is not None else None
    _model_cache = ModelCache(max_models=args.max_resident_models, max_bytes=max_bytes)

    api_key = args.api_key or os.environ.get("MLX_IMAGER_API_KEY", "")
    ImagerAPIHandler._api_key = api_key
    ImagerAPIHandler._auth_health = args.auth_health
    ImagerAPIHandler._auth_metrics = args.auth_metrics
    ImagerAPIHandler._args = args
    if api_key:
        log.info("API key authentication enabled")

    preload: list[str] = list(args.preload or [])
    if args.models_config:
        try:
            entries, aliases = _parse_models_config(args.models_config)
            if aliases:
                _model_aliases.update(aliases)
                log.info("Loaded %d model alias(es)", len(aliases))
            for entry in entries:
                if entry.get("role") == "image" and entry.get("id"):
                    preload.append(entry["id"])
        except Exception as e:
            log.warning("Failed to load models-config: %s", e)

    for m in preload:
        try:
            log.info("Preloading %s …", m)
            model, resolved = _get_image_model(m)
            _model_cache.pin(resolved)
        except Exception as e:
            log.warning("Preload of %s failed: %s", m, e)

    server = ThreadedDualStackHTTPServer((args.host, args.port), ImagerAPIHandler)
    log.info("ai-mlx-imager %s listening on %s:%d (txt2img: %s | edit: %s)",
             _SERVER_VERSION, args.host, args.port, args.model, args.edit_model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
