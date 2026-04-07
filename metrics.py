"""Prometheus metrics for ai-mlx-server."""

# All metrics are created at import time so they're registered globally.
# Import this module only when prometheus_client is available.

from prometheus_client import Counter, Histogram, Gauge, REGISTRY  # noqa: F401

# Request counters
requests_total = Counter(
    "mlx_requests_total",
    "Total HTTP requests",
    ["route", "model", "status"]
)

tokens_in_total = Counter(
    "mlx_tokens_in_total",
    "Total input tokens processed",
    ["model"]
)

tokens_out_total = Counter(
    "mlx_tokens_out_total",
    "Total output tokens generated",
    ["model"]
)

# Histograms
request_duration_seconds = Histogram(
    "mlx_request_duration_seconds",
    "HTTP request duration",
    ["route", "model"]
)

model_load_duration_seconds = Histogram(
    "mlx_model_load_duration_seconds",
    "Time to load a model",
    ["model"]
)

# Gauges (updated on each /metrics scrape from _model_cache)
resident_models = Gauge("mlx_resident_models", "Number of resident models")
resident_bytes = Gauge("mlx_resident_bytes", "Estimated bytes used by resident models")
queue_depth = Gauge("mlx_queue_depth", "Current request queue depth")
