# ai-mlx-server

MLX inference server for Apple Silicon — OpenAI-compatible API with embeddings and LoRA support.

Extracted from [ai-docs](https://github.com/andychoi/ai-docs) for standalone use. Any client that speaks the OpenAI/Ollama API can use this server transparently.

## Features

### Inherited from mlx_lm (no extra code)
- **OpenAI-compatible chat completions** (`POST /v1/chat/completions`) — full streaming SSE support
- **Usage token accounting** — `prompt_tokens` and `completion_tokens` in every response
- **Tool calling** — pass `tools: [...]` in the request body; works with tool-capable models (e.g. Qwen3)

### Added by ai-mlx-server
- **Ollama-compatible API** — `POST /api/generate`, `/api/chat`, `/api/embeddings`; `GET /api/tags`, `/api/version`
- **Batched embeddings** (`POST /v1/embeddings`) — accepts `input: str | list[str]`
- **Multi-model warm pool** — `--preload MODEL` (chat), `--preload-embedding MODEL` (embeddings)
- **LRU model eviction** — `--max-resident-models N`, `--max-resident-gb N`
- **Rich `/health`** — uptime, resident models, RAM stats, queue depth
- **Prometheus `/metrics`** — install with `pip install ai-mlx-server[metrics]`
- **Thinking budget** — pass `thinking_budget` and `enable_thinking` in request body
- **Bearer-token auth** — `--api-key KEY` or `MLX_API_KEY` env var
- **LoRA adapters** — `--adapter-path path/to/adapter`

## Install

### pip (recommended)
```bash
pip install git+https://github.com/andychoi/ai-mlx-server.git
```

### macOS native service (launchd)

Run `ai-mlx-server` automatically at login, with restart-on-crash:

```bash
# Install via pip first, then:
bash packaging/install-service.sh

# Start the service
launchctl load ~/Library/LaunchAgents/com.andychoi.ai-mlx-server.plist

# Stop the service
launchctl unload ~/Library/LaunchAgents/com.andychoi.ai-mlx-server.plist

# View logs
tail -f ~/Library/Logs/ai-mlx-server.log
```

Edit `~/.config/ai-mlx-server/models.yaml` to configure which models to preload at startup.

## Quick Start

```bash
pip install -e .
python server.py --model mlx-community/gemma-4-27b-it-4bit --port 11435
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat inference (OpenAI-compatible, streaming, tools) |
| `POST /v1/embeddings` | Text embeddings (batched) |
| `POST /api/generate` | Ollama generate endpoint |
| `POST /api/chat` | Ollama chat endpoint |
| `POST /api/embeddings` | Ollama embeddings endpoint |
| `GET /api/tags` | List loaded models (Ollama format) |
| `GET /api/version` | Server version |
| `GET /health` | Rich health check (uptime, RAM, queue depth) |
| `GET /metrics` | Prometheus metrics (requires `[metrics]` extra) |

## Configuration

Full list of CLI flags:

```
python server.py [OPTIONS]

Model selection:
  --model MODEL             Default chat model to load (HuggingFace path)
  --adapter-path PATH       LoRA adapter weights directory
  --draft-model MODEL       Speculative decoding draft model

Warm pool / eviction:
  --preload MODEL           Chat model to preload at startup (repeatable)
  --preload-embedding MODEL Embedding model to preload at startup (repeatable)
  --models-config PATH      YAML config listing models to preload
  --max-resident-models N   Maximum number of models to keep in memory
  --max-resident-gb N       Maximum total model memory in GB before LRU eviction

Network:
  --host HOST               Bind address (default: 0.0.0.0)
  --port PORT               Port (default: 11435)

Generation defaults:
  --temp FLOAT              Sampling temperature (default: 0.0)
  --top-p FLOAT             Top-p sampling (default: 1.0)
  --top-k INT               Top-k sampling (default: 0)
  --min-p FLOAT             Min-p sampling (default: 0.0)
  --max-tokens INT          Maximum tokens per response (default: 4096)

Concurrency / cache:
  --decode-concurrency N    Max parallel decode streams (default: 32)
  --prompt-concurrency N    Max parallel prompt evaluations (default: 2)
  --prefill-step-size N     Prefill chunk size (default: 512)
  --prompt-cache-size N     Number of prompt cache entries
  --prompt-cache-bytes N    Total prompt cache size in bytes
  --pipeline               Enable pipeline parallelism

Chat template:
  --chat-template TEMPLATE  Jinja2 chat template string
  --use-default-chat-template
  --chat-template-args JSON Additional template arguments (JSON object)
  --trust-remote-code       Trust remote tokenizer code

Logging:
  --log-level LEVEL         DEBUG | INFO | WARNING | ERROR (default: INFO)
```

## Usage with ai-docs / Ollama clients

Point any Ollama-compatible client at this server:

```bash
# ai-docs
OLLAMA_URL=http://localhost:11435 ./scripts/dev.sh --pg

# Docker containers
OLLAMA_URL=http://host.docker.internal:11435
```

## LoRA Fine-Tuning

```bash
pip install -e ".[lora]"

# Train an adapter
python -m mlx_lm.lora --model mlx-community/gemma-4-27b-it-4bit \
  --data lora/data/ --adapter-path lora/adapters/my-adapter

# Serve with adapter
python server.py --model mlx-community/gemma-4-27b-it-4bit \
  --adapter-path lora/adapters/my-adapter --port 11435
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
