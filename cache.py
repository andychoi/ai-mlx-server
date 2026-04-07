"""Unified LRU model cache for chat and embedding models."""

from __future__ import annotations

import glob
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable

log = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe LRU cache for MLX models.

    Supports max_models (count) and max_bytes (memory) eviction limits.
    Pinned models are exempt from eviction.

    The lock is held only during dict operations, never across model load
    or inference. This prevents deadlocks with mlx_lm's ResponseGenerator.
    """

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

    def get(self, key: str):
        """Return cached value, bumping LRU position. Returns None if not found."""
        import datetime
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            entry = self._cache[key]
            entry["calls"] += 1
            entry["last_used_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return entry["value"]

    def put(self, key: str, value, est_bytes: int = 0, pinned: bool = False, role: str = "chat"):
        """Store value, evicting LRU entries if limits are exceeded."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key]["value"] = value
                self._cache[key]["est_bytes"] = est_bytes
                return
            self._cache[key] = {
                "value": value,
                "est_bytes": est_bytes,
                "pinned": pinned,
                "calls": 0,
                "loaded_at": now,
                "last_used_at": now,
                "role": role,
                "id": key,
            }
            evicted_keys = self._evict_if_needed()
        # Call eviction callbacks outside the lock to avoid deadlock.
        # The callback (e.g. _tear_down_worker) may block on stop_and_join().
        if self._on_evict is not None:
            for evicted_key in evicted_keys:
                try:
                    self._on_evict(evicted_key)
                except Exception as cb_err:
                    log.warning("on_evict callback raised: %s", cb_err)

    def _evict_if_needed(self) -> list[str]:
        """Evict LRU unpinned entries until within limits. Lock must be held.

        Returns list of evicted keys so callers can notify outside the lock.
        """
        evicted: list[str] = []
        while True:
            evictable = [(k, v) for k, v in self._cache.items() if not v["pinned"]]

            over_count = (
                self._max_models is not None
                and len(self._cache) > self._max_models
            )
            over_bytes = (
                self._max_bytes is not None
                and sum(v["est_bytes"] for v in self._cache.values()) > self._max_bytes
            )

            if not (over_count or over_bytes):
                break

            if not evictable:
                log.warning("Cannot evict: all %d models are pinned", len(self._cache))
                break

            # Evict LRU (first in OrderedDict among evictable)
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
            evicted.append(oldest_key)
        return evicted

    def stats(self) -> list[dict]:
        """Return list of dicts for each resident model (for /health)."""
        with self._lock:
            return [
                {
                    "id": v["id"],
                    "role": v.get("role", "chat"),
                    "size_gb": round(v["est_bytes"] / 1e9, 2),
                    "loaded_at": v["loaded_at"],
                    "last_used_at": v["last_used_at"],
                    "calls": v["calls"],
                    "pinned": v["pinned"],
                }
                for v in self._cache.values()
            ]

    def total_bytes(self) -> int:
        with self._lock:
            return sum(v["est_bytes"] for v in self._cache.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


def estimate_model_bytes(model_path: str) -> int:
    """Estimate model size by summing *.safetensors files in HF cache."""
    safe_path = model_path.replace("/", "--")
    pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{safe_path}/snapshots/*/*.safetensors"
    )
    total = sum(os.path.getsize(p) for p in glob.glob(pattern) if os.path.isfile(p))
    return total if total > 0 else 0
