"""In-process MLX inference — for direct Python use without the HTTP server.

Use this when you need sub-millisecond latency or direct access to model
internals (e.g. logits, attention weights, custom decoding).

For most use cases, prefer the HTTP server (server.py) which handles
concurrency, model caching, and provides a standard API.
"""

from __future__ import annotations

import logging
import os
import re
import threading

from cache import ModelCache

_log = logging.getLogger(__name__)

# Module-level LRU caches — models loaded once per process.
_mlx_model_cache: ModelCache = ModelCache()
_mlx_embed_cache: ModelCache = ModelCache()

# Serialize all MLX GPU access — Metal command buffers are not thread-safe.
_mlx_lock = threading.Lock()


def _load_hf_offline_first(load_fn, model_path: str, lib_name: str):
    """Try loading a HuggingFace model offline first; fall back to online."""
    orig = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return load_fn(model_path)
    except Exception:
        _log.info("Model %s not in local cache, downloading via %s …",
                   model_path, lib_name)
        if orig is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = orig
        return load_fn(model_path)
    finally:
        if orig is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = orig


def embed(model_path: str, text: str) -> list[float]:
    """Get embedding vector via mlx-embeddings. Returns list of floats.

    Requires: pip install mlx-embeddings
    Model cache: loaded once per process.
    """
    try:
        from mlx_embeddings import load as _mlx_embed_load
    except ImportError:
        raise RuntimeError(
            "mlx-embeddings not installed. Run: pip install mlx-embeddings"
        )

    with _mlx_lock:
        cached = _mlx_embed_cache.get(model_path)
        if cached is None:
            model, tokenizer = _load_hf_offline_first(
                _mlx_embed_load, model_path, "mlx-embeddings"
            )
            _mlx_embed_cache.put(model_path, (model, tokenizer), role="embedding")
            cached = (model, tokenizer)
        model, tokenizer = cached

        try:
            input_ids = tokenizer.encode(text, return_tensors="mlx")
            outputs = model(input_ids)
            return outputs.text_embeds[0].tolist()
        except Exception as e:
            raise RuntimeError(f"MLX embedding failed (model={model_path}): {e}") from e


def invoke(
    model_path: str,
    prompt: str | list[dict],
    max_tokens: int = 4096,
    thinking_budget: int = 0,
    enable_thinking: bool = True,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, int, int]:
    """Invoke a local MLX model. Returns (text, tokens_in, tokens_out).

    Tries mlx_lm first (text-only), then mlx_vlm (vision-language).
    Models are cached in-process.
    """
    # Try mlx_lm (text-only models)
    _mlx_lm_available = False
    try:
        from mlx_lm import load as _lm_load, generate as _lm_generate
        _mlx_lm_available = True
    except ImportError:
        pass

    # Fall back to mlx_vlm (vision-language models)
    _mlx_vlm_available = False
    if not _mlx_lm_available:
        try:
            from mlx_vlm import load as _vlm_load, generate as _vlm_generate
            from mlx_vlm.prompt_utils import apply_chat_template as _apply_chat_template
            from mlx_vlm.utils import load_config as _load_vlm_config
            _mlx_vlm_available = True
        except ImportError:
            pass

    if not _mlx_lm_available and not _mlx_vlm_available:
        raise RuntimeError(
            "No MLX library found. Install one of:\n"
            "  pip install mlx-lm      # text models (recommended)\n"
            "  pip install mlx-vlm     # vision-language models"
        )

    with _mlx_lock:
        try:
            if _mlx_lm_available:
                cached_lm = _mlx_model_cache.get(model_path)
                if cached_lm is None:
                    model, tokenizer = _load_hf_offline_first(
                        _lm_load, model_path, "mlx-lm"
                    )
                    _mlx_model_cache.put(model_path, (model, tokenizer, "lm"), role="chat")
                    cached_lm = (model, tokenizer, "lm")

                model, tokenizer, _ = cached_lm

                if hasattr(tokenizer, "apply_chat_template"):
                    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
                    tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
                    if not enable_thinking:
                        tpl_kwargs["enable_thinking"] = False
                    if chat_template_kwargs:
                        tpl_kwargs.update(chat_template_kwargs)
                    try:
                        formatted = tokenizer.apply_chat_template(messages, **tpl_kwargs)
                    except TypeError:
                        formatted = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                else:
                    formatted = prompt

                output = _lm_generate(model, tokenizer, prompt=formatted,
                                      max_tokens=max_tokens, verbose=False)

            else:  # mlx_vlm
                cached_vlm = _mlx_model_cache.get(model_path)
                if cached_vlm is None:
                    model, processor = _load_hf_offline_first(
                        _vlm_load, model_path, "mlx-vlm"
                    )
                    cfg = _load_vlm_config(model_path)
                    _mlx_model_cache.put(model_path, (model, processor, cfg), role="chat")
                    cached_vlm = (model, processor, cfg)

                model, processor, mlx_cfg = cached_vlm
                formatted = _apply_chat_template(processor, mlx_cfg, prompt, num_images=0)

                kwargs: dict = {"max_tokens": max_tokens, "verbose": False}
                if thinking_budget > 0:
                    kwargs["thinking_budget"] = thinking_budget

                output = _vlm_generate(model, processor, formatted, **kwargs)

            prompt_text = " ".join(m.get("content", "") for m in prompt) if isinstance(prompt, list) else prompt
            tok_in = len(prompt_text) // 4
            tok_out_raw = len(output) // 4

            # Strip <think>…</think> reasoning blocks
            text = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
            text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
            text = text.strip()

            return text, tok_in, tok_out_raw

        except Exception as e:
            raise RuntimeError(f"MLX invocation failed (model={model_path}): {e}") from e
