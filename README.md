# ai-mlx-server

MLX inference server for Apple Silicon — OpenAI-compatible API with embeddings and LoRA support.

Extracted from [ai-docs](https://github.com/andychoi/ai-docs) for standalone use. Any client that speaks the OpenAI/Ollama API can use this server transparently.

## Quick Start

```bash
pip install -e .
python server.py --model mlx-community/gemma-4-27b-it-4bit --port 8085
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat inference (OpenAI-compatible) |
| `POST /v1/embeddings` | Text embeddings |
| `GET /health` | Health check |

## Usage with ai-docs / Ollama clients

Point any Ollama-compatible client at this server:

```bash
# ai-docs
OLLAMA_URL=http://localhost:8085 ./scripts/dev.sh --pg

# Docker containers
OLLAMA_URL=http://host.docker.internal:8085
```

## LoRA Fine-Tuning

```bash
pip install -e ".[lora]"

# Train an adapter
python -m mlx_lm.lora --model mlx-community/gemma-4-27b-it-4bit \
  --data lora/data/ --adapter-path lora/adapters/my-adapter

# Serve with adapter
python server.py --model mlx-community/gemma-4-27b-it-4bit \
  --adapter-path lora/adapters/my-adapter --port 8085
```

## In-Process Use

For direct Python access without the HTTP server (sub-millisecond latency, custom decoding):

```python
from invoke import invoke, embed

text, tok_in, tok_out = invoke("mlx-community/gemma-4-27b-it-4bit", "Hello!")
vector = embed("mlx-community/snowflake-arctic-embed-l-v2.0-4bit", "some text")
```

## Why separate from Ollama?

Ollama now supports MLX backend and is the recommended default. Use this server when you need:

- **LoRA adapters** — load fine-tuned adapters at serving time
- **Custom embeddings** — use mlx-embeddings models not available in Ollama
- **Vision-language models** — mlx_vlm support
- **Direct model access** — logits, attention weights, custom decoding via `invoke.py`
- **Maximum throughput** — fine-grained concurrency control for agent workloads
