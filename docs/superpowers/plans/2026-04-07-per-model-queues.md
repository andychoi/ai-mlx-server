# Per-model ResponseGenerator workers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every chat model its own `ResponseGenerator` (and `LRUPromptCache`), created lazily on first request and torn down on LRU eviction, so requests to different models queue independently and do not share a prompt cache.

**Architecture:** A module-level `_model_workers: dict[str, ResponseGenerator]` registry is managed by `_get_or_create_worker()` and `_tear_down_worker()`. `MLXAPIHandler.do_POST` and `_handle_ollama_chat` peek the `model` field and temporarily swap `self.response_generator` before calling the parent handler. `ModelCache` gains an `on_evict` callback so eviction automatically tears down the matching worker. `/health` and `/metrics` are updated to report per-model queue depths using the correct `ResponseGenerator.requests` attribute (the previous `_queue` name was wrong).

**Tech Stack:** Python 3.11+, `mlx_lm.server.ResponseGenerator`, `mlx_lm.server.ModelProvider`, `mlx_lm.server.LRUPromptCache`, `threading.Lock`, stdlib `queue.Queue`

---

## File map

| File | Change |
|---|---|
| `cache.py` | Add `on_evict: Callable[[str], None] \| None` param; call it in `_evict_if_needed` |
| `server.py` | Module-level `_model_workers`, `_workers_lock`, `_server_args`; add `_get_or_create_worker`, `_tear_down_worker`; route `do_POST` + `_handle_ollama_chat`; register eviction callback; eager workers for `--preload`; fix `/health` + `/metrics` queue depth; tear down all workers on shutdown |
| `tests/test_cache.py` | Add one test: `on_evict` callback fires with correct key |
| `tests/test_per_model_workers.py` | New file: worker creation, reuse, teardown, fallback routing |

---

## Task 1 — `on_evict` callback in `ModelCache`

**Files:** Modify `cache.py:24-66`, Modify `tests/test_cache.py`

- [ ] **Step 1.1 — Write the failing test**

Append to `tests/test_cache.py` (inside `TestModelCacheLRU` class, before `if __name__ == "__main__"`):

```python
def test_on_evict_callback_fires(self):
    """on_evict is called with the evicted key when LRU eviction occurs."""
    evicted = []
    cache = ModelCache(max_models=2, on_evict=evicted.append)
    cache.put("m0", "v0")
    cache.put("m1", "v1")
    cache.put("m2", "v2")   # triggers eviction of m0
    self.assertEqual(evicted, ["m0"])
    cache.put("m3", "v3")   # triggers eviction of m1
    self.assertEqual(evicted, ["m0", "m1"])
```

- [ ] **Step 1.2 — Run test to confirm it fails**

```bash
cd /Users/andymini/ai/ai-mlx-server
python -m pytest tests/test_cache.py::TestModelCacheLRU::test_on_evict_callback_fires -v
```

Expected: `FAILED` — `TypeError: __init__() got an unexpected keyword argument 'on_evict'`

- [ ] **Step 1.3 — Add `on_evict` to `ModelCache.__init__`**

In `cache.py`, change the `__init__` signature (currently line 24):

```python
# BEFORE
def __init__(
    self,
    max_models: int | None = None,
    max_bytes: int | None = None,
):
    self._max_models = max_models
    self._max_bytes = max_bytes
    self._cache: OrderedDict[str, dict] = OrderedDict()
    self._lock = threading.Lock()
```

```python
# AFTER
from collections.abc import Callable   # add to top of file imports

def __init__(
    self,
    max_models: int | None = None,
    max_bytes: int | None = None,
    on_evict: Callable[[str], None] | None = None,
):
    self._max_models = max_models
    self._max_bytes = max_bytes
    self._on_evict = on_evict
    self._cache: OrderedDict[str, dict] = OrderedDict()
    self._lock = threading.Lock()
```

- [ ] **Step 1.4 — Call the callback in `_evict_if_needed`**

In `cache.py`, update `_evict_if_needed` (currently lines 68-100). Replace the eviction block:

```python
# BEFORE (inside the while loop, after oldest_key is chosen)
            oldest_key = evictable[0][0]
            entry = self._cache.pop(oldest_key)
            log.info("Evicting model %s (est %.1f GB)", oldest_key, entry["est_bytes"] / 1e9)
            # Release reference so Python GC can collect the model
            del entry["value"]
            # Clear Metal cache to actually free VRAM
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception:
                pass
```

```python
# AFTER
            oldest_key = evictable[0][0]
            entry = self._cache.pop(oldest_key)
            log.info("Evicting model %s (est %.1f GB)", oldest_key, entry["est_bytes"] / 1e9)
            # Notify caller before releasing the model reference
            if self._on_evict is not None:
                try:
                    self._on_evict(oldest_key)
                except Exception as cb_err:
                    log.warning("on_evict callback raised: %s", cb_err)
            # Release reference so Python GC can collect the model
            del entry["value"]
            # Clear Metal cache to actually free VRAM
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception:
                pass
```

- [ ] **Step 1.5 — Run test to confirm it passes**

```bash
python -m pytest tests/test_cache.py -v
```

Expected: all tests pass (was 6, now 7).

- [ ] **Step 1.6 — Commit**

```bash
git add cache.py tests/test_cache.py
git commit -m "feat: add on_evict callback to ModelCache

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2 — Worker registry and lifecycle functions

**Files:** Modify `server.py`, Create `tests/test_per_model_workers.py`

- [ ] **Step 2.1 — Write failing tests**

Create `tests/test_per_model_workers.py`:

```python
"""Unit tests for per-model ResponseGenerator worker registry.

These tests mock ResponseGenerator so no MLX model is required.
"""
import sys
import os
import threading
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_mock_rg():
    """Return a mock ResponseGenerator with a real queue-like requests attribute."""
    import queue
    rg = MagicMock()
    rg.requests = queue.Queue()
    return rg


class TestWorkerRegistry(unittest.TestCase):

    def setUp(self):
        """Reset the worker registry before each test."""
        import server
        with server._workers_lock:
            server._model_workers.clear()

    def test_get_or_create_worker_creates_on_first_call(self):
        """First call for a model creates and stores a new worker."""
        import server
        mock_rg = _make_mock_rg()

        with patch("server.ResponseGenerator", return_value=mock_rg), \
             patch("server.ModelProvider"), \
             patch("server.LRUPromptCache"):
            worker = server._get_or_create_worker("test-model")

        self.assertIs(worker, mock_rg)
        self.assertIn("test-model", server._model_workers)

    def test_get_or_create_worker_returns_same_worker_on_second_call(self):
        """Second call for the same model returns the existing worker without creating a new one."""
        import server
        mock_rg = _make_mock_rg()

        with patch("server.ResponseGenerator", return_value=mock_rg) as rg_cls, \
             patch("server.ModelProvider"), \
             patch("server.LRUPromptCache"):
            w1 = server._get_or_create_worker("test-model")
            w2 = server._get_or_create_worker("test-model")

        self.assertIs(w1, w2)
        self.assertEqual(rg_cls.call_count, 1)

    def test_tear_down_worker_removes_from_registry(self):
        """_tear_down_worker removes the model from _model_workers."""
        import server
        mock_rg = _make_mock_rg()
        with server._workers_lock:
            server._model_workers["my-model"] = mock_rg

        server._tear_down_worker("my-model")

        self.assertNotIn("my-model", server._model_workers)
        mock_rg.stop_and_join.assert_called_once()

    def test_tear_down_worker_noop_for_unknown_model(self):
        """_tear_down_worker is a no-op when the model has no worker."""
        import server
        # Should not raise
        server._tear_down_worker("nonexistent-model")

    def test_get_or_create_worker_uses_server_args(self):
        """Worker creation uses _server_args to build the ModelProvider."""
        import server
        import argparse
        fake_args = argparse.Namespace(
            model=None, trust_remote_code=False, chat_template=None,
            use_default_chat_template=False, adapter_path=None, draft_model=None,
            num_draft_tokens=3, temp=0.0, top_p=1.0, top_k=0, min_p=0.0,
            max_tokens=4096, chat_template_args={}, decode_concurrency=32,
            prompt_concurrency=2, prefill_step_size=512, prompt_cache_size=None,
            prompt_cache_bytes=None, pipeline=False,
        )
        server._server_args = fake_args

        mock_rg = _make_mock_rg()
        captured_provider_args = []

        def capture_mp(a):
            captured_provider_args.append(a.model)
            return MagicMock()

        with patch("server.ResponseGenerator", return_value=mock_rg), \
             patch("server.ModelProvider", side_effect=capture_mp), \
             patch("server.LRUPromptCache"):
            server._get_or_create_worker("some/model")

        self.assertEqual(captured_provider_args, ["some/model"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2.2 — Run tests to confirm they fail**

```bash
python -m pytest tests/test_per_model_workers.py -v
```

Expected: `FAILED` — `AttributeError: module 'server' has no attribute '_model_workers'`

- [ ] **Step 2.3 — Add registry globals and lifecycle functions to `server.py`**

After the `_embed_lock` declaration (currently around line 52), add:

```python
# ── Per-model ResponseGenerator worker registry ─────────────────────────────
# Each chat model gets its own ResponseGenerator (and LRUPromptCache) so that
# requests to different models queue independently.
_model_workers: dict[str, "ResponseGenerator"] = {}
_workers_lock = threading.Lock()
# Populated by main() so worker creation can build ModelProvider instances.
_server_args = None
```

After the `_embed_lock` block and new registry block, add these two functions:

```python
def _get_or_create_worker(model_id: str) -> "ResponseGenerator":
    """Return the per-model ResponseGenerator, creating it lazily if needed.

    Thread-safe: concurrent calls for the same model will only create one worker.
    Uses _server_args to build ModelProvider; must be called after main() sets it.
    """
    with _workers_lock:
        if model_id in _model_workers:
            return _model_workers[model_id]

    # Create outside lock to avoid holding it during potentially slow setup.
    import copy
    args_copy = copy.copy(_server_args)
    args_copy.model = model_id
    mp = ModelProvider(args_copy)
    pc = LRUPromptCache((_server_args.prompt_cache_size or 1))
    worker = ResponseGenerator(mp, pc)

    with _workers_lock:
        # Re-check: another thread may have created it while we were building.
        if model_id not in _model_workers:
            _model_workers[model_id] = worker
        else:
            # Discard ours; stop the extra thread we started.
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
```

- [ ] **Step 2.4 — Run tests to confirm they pass**

```bash
python -m pytest tests/test_per_model_workers.py -v
python -m pytest tests/ -v
```

Expected: all 5 new tests pass; all prior tests still pass.

- [ ] **Step 2.5 — Commit**

```bash
git add server.py tests/test_per_model_workers.py
git commit -m "feat: per-model worker registry with lazy creation and teardown

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3 — Route requests to per-model workers in `do_POST` and `_handle_ollama_chat`

**Files:** Modify `server.py:362-386` (do_POST), Modify `server.py:_handle_ollama_chat`

- [ ] **Step 3.1 — Update `do_POST` to route `/v1/chat/completions` through the per-model worker**

Find the current `do_POST` method (around line 362). Replace the `/v1/chat/completions` block:

```python
# BEFORE
    if self.path == "/v1/chat/completions":
        # Peek at body for thinking params
        body = self._peek_body()
        if body and ("thinking_budget" in body or "enable_thinking" in body):
            self._handle_thinking_completion(body)
            return
    # Delegate chat/completions (and /v1/completions) to parent
    super().do_POST()
```

```python
# AFTER
    if self.path == "/v1/chat/completions":
        # Peek at body for thinking params and model routing (single read).
        body = self._peek_body()
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
```

- [ ] **Step 3.2 — Add `_dispatch_to_worker` helper method to `MLXAPIHandler`**

Add this method to `MLXAPIHandler` (after `_peek_body`, before `_handle_thinking_completion`):

```python
def _dispatch_to_worker(self, model_id: str) -> None:
    """Swap self.response_generator to the per-model worker and call parent."""
    worker = _get_or_create_worker(model_id)
    original_rg = self.response_generator
    self.response_generator = worker
    try:
        super().do_POST()
    finally:
        self.response_generator = original_rg
```

- [ ] **Step 3.3 — Route Ollama chat through per-model worker**

In `_handle_ollama_chat`, find the section that swaps `self.path`, `self.rfile`, and `self.wfile` and calls `super().do_POST()`. The model is available as `model = ollama_body.get("model", self._default_model)` (already computed earlier in that method).

Replace the two `super().do_POST()` calls inside `_handle_ollama_chat` with `self._dispatch_to_worker(model)`:

```python
# BEFORE (streaming branch)
                self.wfile = _StreamTranslatorWfile(original_wfile, model)
                super().do_POST()
                # Flush any remaining buffered bytes
                self.wfile.flush()
```

```python
# AFTER (streaming branch)
                self.wfile = _StreamTranslatorWfile(original_wfile, model)
                if _server_args is not None and model:
                    self._dispatch_to_worker(model)
                else:
                    super().do_POST()
                # Flush any remaining buffered bytes
                self.wfile.flush()
```

```python
# BEFORE (non-streaming branch)
                buf = io.BytesIO()
                self.wfile = buf
                super().do_POST()
```

```python
# AFTER (non-streaming branch)
                buf = io.BytesIO()
                self.wfile = buf
                if _server_args is not None and model:
                    self._dispatch_to_worker(model)
                else:
                    super().do_POST()
```

- [ ] **Step 3.4 — Verify import and tests still pass**

```bash
python -c "import server"
python -m pytest tests/ -v
```

Expected: clean import, all tests pass.

- [ ] **Step 3.5 — Commit**

```bash
git add server.py
git commit -m "feat: route chat requests to per-model ResponseGenerator workers

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4 — Wire `_server_args`, eviction callback, and eager startup workers

**Files:** Modify `server.py:main()` (lines ~720-821)

- [ ] **Step 4.1 — Store `_server_args` and register eviction callback at startup**

In `main()`, right after `args = parser.parse_args()`, add:

```python
    # Store args globally so _get_or_create_worker can build ModelProvider instances.
    global _server_args
    _server_args = args
```

Change the `_model_cache` construction (currently around line 727) to register the eviction callback:

```python
# BEFORE
    _model_cache = ModelCache(
        max_models=args.max_resident_models,
        max_bytes=max_bytes,
    )
```

```python
# AFTER
    _model_cache = ModelCache(
        max_models=args.max_resident_models,
        max_bytes=max_bytes,
        on_evict=_tear_down_worker,
    )
```

- [ ] **Step 4.2 — Create workers eagerly for `--preload` chat models**

After the preload loop (currently around line 783, after `_loaded_model_names.add(model_id)`), add eager worker creation:

```python
    # Preload chat models sequentially
    for model_id in preload_chat:
        log.info("Preloading chat model: %s", model_id)
        t0 = time.time()
        try:
            orig_model = args.model
            args.model = model_id
            model_provider_tmp = ModelProvider(args)
            model_provider_tmp.load()
            args.model = orig_model
            est = estimate_model_bytes(model_id)
            _model_cache.put(model_id, model_provider_tmp, est_bytes=est, pinned=True, role="chat")
            _loaded_model_names.add(model_id)
            log.info("Preloaded chat model %s in %.1fs", model_id, time.time() - t0)
            # Create the per-model worker eagerly so it's warm before first request.
            _get_or_create_worker(model_id)
            log.info("Worker ready for %s", model_id)
        except Exception as e:
            log.error("Failed to preload chat model %s: %s", model_id, e)
```

- [ ] **Step 4.3 — Tear down all workers on shutdown**

In the `except KeyboardInterrupt` block at the bottom of `main()`:

```python
# BEFORE
    except KeyboardInterrupt:
        log.info("Shutting down…")
        response_generator.stop_and_join()
        server.server_close()
```

```python
# AFTER
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
```

- [ ] **Step 4.4 — Verify**

```bash
python -c "import server"
python server.py --help
python -m pytest tests/ -v
```

Expected: clean import, help shows all flags, all tests pass.

- [ ] **Step 4.5 — Commit**

```bash
git add server.py
git commit -m "feat: wire eviction callback and eager startup workers for --preload models

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5 — Fix `/health` and `/metrics` queue depths

**Files:** Modify `server.py:_handle_health`, `server.py:_handle_metrics`

- [ ] **Step 5.1 — Fix `_handle_health`**

Replace the broken `_queue` lookup in `_handle_health`:

```python
# BEFORE
        # Try to read queue depth from ResponseGenerator
        q = getattr(getattr(MLXAPIHandler, '_response_generator', None), '_queue', None)
        if q is not None:
            queue_depth = q.qsize()
        else:
            log.debug("queue_depth unavailable: _response_generator has no _queue attribute")
            queue_depth = 0

        self._json_response(200, {
            "status": "ok",
            "version": _SERVER_VERSION,
            "uptime_seconds": int(time.time() - _server_start_time),
            "resident_models": _model_cache.stats(),
            "ram_used_gb": ram_used_gb,
            "ram_available_gb": ram_available_gb,
            "queue_depth": queue_depth,
        })
```

```python
# AFTER
        # Per-model queue depths (ResponseGenerator.requests is the correct attribute).
        with _workers_lock:
            workers_snapshot = dict(_model_workers)
        queue_depths = {
            mid: w.requests.qsize()
            for mid, w in workers_snapshot.items()
        }
        # Include the default (non-worker) generator if present.
        default_rg = getattr(MLXAPIHandler, '_response_generator', None)
        default_q = getattr(default_rg, 'requests', None)
        if default_q is not None and not queue_depths:
            queue_depths["_default"] = default_q.qsize()

        self._json_response(200, {
            "status": "ok",
            "version": _SERVER_VERSION,
            "uptime_seconds": int(time.time() - _server_start_time),
            "resident_models": _model_cache.stats(),
            "ram_used_gb": ram_used_gb,
            "ram_available_gb": ram_available_gb,
            "queue_depths": queue_depths,
        })
```

- [ ] **Step 5.2 — Fix `_handle_metrics`**

Replace the broken `_queue` lookup in `_handle_metrics`:

```python
# BEFORE
        # Update queue gauge
        q = getattr(getattr(MLXAPIHandler, '_response_generator', None), '_queue', None)
        if q is not None:
            _queue_depth_gauge.set(q.qsize())
```

```python
# AFTER
        # Sum queue depths across all per-model workers.
        with _workers_lock:
            workers_snapshot = dict(_model_workers)
        total_queued = sum(w.requests.qsize() for w in workers_snapshot.values())
        default_rg = getattr(MLXAPIHandler, '_response_generator', None)
        default_q = getattr(default_rg, 'requests', None)
        if default_q is not None and not workers_snapshot:
            total_queued += default_q.qsize()
        _queue_depth_gauge.set(total_queued)
```

- [ ] **Step 5.3 — Verify**

```bash
python -c "import server"
python -m pytest tests/ -v
```

Expected: clean import, all tests pass.

- [ ] **Step 5.4 — Commit and push**

```bash
git add server.py
git commit -m "fix: per-model queue depths in /health and /metrics

- Replace broken _queue lookup with ResponseGenerator.requests.qsize()
- /health returns queue_depths dict keyed by model_id
- /metrics mlx_queue_depth gauge sums across all workers

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git push origin main
```

---

## Self-review

**Spec coverage:**
- ✅ `on_evict` callback on `ModelCache` — Task 1
- ✅ `_model_workers` registry + `_get_or_create_worker` + `_tear_down_worker` — Task 2
- ✅ Request routing in `do_POST` + `_handle_ollama_chat` — Task 3
- ✅ Eager workers for `--preload` models — Task 4
- ✅ Eviction callback registered on `_model_cache` — Task 4
- ✅ `_server_args` stored at startup — Task 4
- ✅ Shutdown tears down all workers — Task 4
- ✅ `/health` queue depths fixed — Task 5
- ✅ `/metrics` gauge fixed — Task 5
- ✅ Tests for `on_evict` callback — Task 1
- ✅ Tests for worker creation/reuse/teardown/fallback — Task 2

**Placeholder scan:** No TBDs or incomplete sections found.

**Type consistency:**
- `_get_or_create_worker(model_id: str) -> ResponseGenerator` — used consistently in Tasks 2, 3, 4
- `_tear_down_worker(model_id: str)` — used consistently in Tasks 2, 4
- `_dispatch_to_worker(model_id: str)` — defined in Task 3, used in Tasks 3 (do_POST + _handle_ollama_chat)
- `worker.requests.qsize()` — correct attribute name (not `_queue`) in Task 5
