# Design: Per-model ResponseGenerator workers

**Date:** 2026-04-07  
**Repo:** `andychoi/ai-mlx-server`  
**Branch target:** `main`

---

## Problem

The current server has a single `ResponseGenerator` shared across all chat/completion requests. This means:

1. **No queue isolation** — a long generation on `gemma4:26b` blocks all queued requests to `gemma4:e4b`, even though they are independent models.
2. **Shared `LRUPromptCache`** — entries from model A crowd out KV-cache entries for model B, reducing prompt-cache hit rate and degrading response quality for both.
3. **`/health` queue depth is broken** — code looks for `_response_generator._queue` but the actual attribute is `ResponseGenerator.requests`.

---

## Goal

Each model gets its own `ResponseGenerator` with its own `LRUPromptCache`. Workers are created lazily on first request and torn down when the model is LRU-evicted. `--preload` models get workers created eagerly at startup.

---

## Architecture

### Worker registry (`server.py`)

```python
_model_workers: dict[str, ResponseGenerator] = {}
_workers_lock = threading.Lock()
```

Two functions manage the registry:

**`_get_or_create_worker(model_id: str) -> ResponseGenerator`**
1. Acquire `_workers_lock`
2. Return existing worker if present
3. Otherwise: create `ModelProvider` for this model, create `LRUPromptCache`, create `ResponseGenerator`, store in `_model_workers`, return it

**`_tear_down_worker(model_id: str)`**
1. Acquire `_workers_lock`
2. Pop worker from `_model_workers` (no-op if missing)
3. Drain `worker.requests` queue — for each pending item, send a 503 response via its result callback
4. Call `worker.stop_and_join()`

`_tear_down_worker` is registered as the `on_evict` callback on `ModelCache` so eviction and worker teardown are always coupled.

### Eviction callback (`cache.py`)

`ModelCache.__init__` gains an optional parameter:

```python
def __init__(
    self,
    max_models: int | None = None,
    max_bytes: int | None = None,
    on_evict: Callable[[str], None] | None = None,
):
```

`_evict_if_needed()` calls `on_evict(key)` before deleting the cache entry. This gives the caller a chance to tear down any associated resources (worker thread, prompt cache) before the model object is released.

### Request routing (`MLXAPIHandler`)

In `do_POST` for `/v1/chat/completions` and in `_handle_ollama_chat`:

1. Peek at the `model` field from the request body
2. Call `_get_or_create_worker(model_id)`
3. Temporarily set `self.response_generator = worker`
4. Dispatch to `super().do_POST()` (or the existing Ollama adapter path)
5. Restore original `self.response_generator` in `finally`

This is safe because `HTTPServer` creates a fresh handler instance per connection — `self.response_generator` is not shared between concurrent requests.

If the `model` field is absent, fall back to `_default_model`; if that is also empty, use the original `response_generator` unchanged (preserves existing behaviour for bare `--model` usage).

### `--preload` eager startup

In `main()`, after the model preload loop, for each preloaded chat model call `_get_or_create_worker(model_id)` immediately. This ensures warm models have their workers ready before the first request arrives, with no lazy-creation latency.

---

## `/health` fix

Replace the broken `_response_generator._queue` lookup with per-model queue depths:

```json
"queue_depths": {
  "mlx-community/gemma-4-27b-4bit": 0,
  "mlx-community/gemma-4-e4b-4bit": 2
}
```

Remove the top-level `queue_depth` scalar (it was always 0 due to the wrong attribute name). `/metrics` `mlx_queue_depth` gauge becomes the sum across all workers.

---

## Edge cases

| Situation | Behaviour |
|---|---|
| `model` field absent | Use `_default_model`; fall back to original `response_generator` |
| Model evicted while request in flight | In-flight request finishes on existing worker; eviction waits for `stop_and_join` after draining the queue |
| Race: eviction + new request for same model | `_workers_lock` serialises: either new request gets the existing worker, or it creates a fresh one after eviction completes |
| `_handle_thinking_completion` (invoke.invoke path) | No `ResponseGenerator` involved — unaffected |
| Embedding models | Handled by `_embed_lock` path — unaffected |

---

## Files changed

| File | Change |
|---|---|
| `cache.py` | Add `on_evict` callback to `ModelCache.__init__`; call it in `_evict_if_needed` |
| `server.py` | `_model_workers` + `_workers_lock`; `_get_or_create_worker()`; `_tear_down_worker()`; routing in `do_POST` + `_handle_ollama_chat`; eager worker creation for `--preload` models; fix `/health` + `/metrics` queue depth |
| `tests/test_cache.py` | Add: eviction triggers `on_evict` callback with correct key |
| `tests/test_per_model_workers.py` | New: worker created on first request; same worker returned on second request; eviction calls `_tear_down_worker`; fallback to default when model field absent |

---

## What is NOT changing

- Embedding pipeline (`_embed_lock`, `_get_embed_model`) — no change
- `invoke.py` / `_handle_thinking_completion` — no change
- Auth, metrics collection, Ollama adapter field translation — no change
- Metal serialises GPU execution regardless — this design improves queue isolation and prompt-cache quality, not raw throughput
