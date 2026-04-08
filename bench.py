#!/usr/bin/env python3
"""Interactive benchmark: Qwen3 vs Gemma3 on ai-mlx-server features.

Tests baseline accuracy, thinking-control features, and bare-tag resolution
against a running server, then prints a side-by-side comparison table.

Usage:
    python bench.py
    python bench.py --models gemma-4-e4b-it-4bit mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit mlx-community/Qwen3.5-27B-MLX-4bit
    python bench.py --server http://localhost:11435 --skip-embed

Health check:
    curl -s http://localhost:11435/health | python3 -m json.tool    
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Callable

import requests

# ── ANSI colours (gracefully disabled when not a tty) ────────────────────────
_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

DIM   = lambda t: _c("2",      t)
BOLD  = lambda t: _c("1",      t)
GREEN = lambda t: _c("32",     t)
RED   = lambda t: _c("31",     t)
CYAN  = lambda t: _c("36",     t)
YELLOW= lambda t: _c("33",     t)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SERVER = "http://localhost:11435"
DEFAULT_MODELS = ["Qwen3-1.7B-4bit", "gemma-3-1b-it-bf16"]
# Models known to support thinking templates (chat_template_kwargs / enable_thinking)
THINKING_MODELS = {"qwen", "qwq", "gemma"}
# Models that also support the /no_think soft-switch in the prompt/system message.
# Qwen3.5 removed this — only chat_template_kwargs works there.
SOFT_SWITCH_MODELS = {"qwen3-", "qwq"}

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _supports_thinking(model: str) -> bool:
    low = model.lower()
    return any(k in low for k in THINKING_MODELS)

def _supports_soft_switch(model: str) -> bool:
    """Qwen3 supports /no_think prefix; Qwen3.5+ and Gemma do not."""
    low = model.lower()
    return any(k in low for k in SOFT_SWITCH_MODELS)


def stream_chat(
    server: str,
    model: str,
    messages: list[dict],
    options: dict | None = None,
    **extra,
) -> tuple[str, float, float, float]:
    """POST /api/chat with stream=True.

    Returns (full_text, ttft_ms, total_ms, approx_words_per_sec).
    Prints tokens live in dim colour.
    """
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        **({"options": options} if options else {}),
        **extra,
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []

    try:
        resp = requests.post(
            f"{server}/api/chat",
            json=body,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                print(DIM(token), end="", flush=True)
                chunks.append(token)
            if chunk.get("done"):
                break

    except requests.RequestException as e:
        print(RED(f"\n[request failed: {e}]"))
        return "", 0.0, 0.0, 0.0

    print()  # newline after streamed tokens
    t1 = time.perf_counter()
    full_text = "".join(chunks)
    total_ms = (t1 - t0) * 1000
    total_secs = total_ms / 1000
    wps = len(full_text.split()) / total_secs if total_secs > 0 else 0.0
    return full_text, ttft or total_ms, total_ms, wps


def one_shot(
    server: str,
    model: str,
    prompt: str,
    system: str | None = None,
    options: dict | None = None,
    **extra,
) -> tuple[str, float, float, float]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return stream_chat(server, model, messages, options=options, **extra)


# ── Result tracking ───────────────────────────────────────────────────────────

class Result:
    __slots__ = ("model", "task", "ttft", "total_ms", "wps", "passed", "skipped")

    def __init__(self, model: str, task: str):
        self.model = model
        self.task = task
        self.ttft = 0.0
        self.total_ms = 0.0
        self.wps = 0.0
        self.passed: bool | None = None  # None = skipped
        self.skipped = False


def _run(
    results: list[Result],
    server: str,
    model: str,
    task: str,
    prompt: str,
    check: Callable[[str], bool],
    system: str | None = None,
    options: dict | None = None,
    skip: bool = False,
    **extra,
) -> None:
    r = Result(model, task)
    results.append(r)
    label = f"  {CYAN(model)} / {BOLD(task)}"

    if skip:
        print(f"{label}  {YELLOW('SKIP')}")
        r.skipped = True
        return

    print(f"{label}")
    text, ttft, total_ms, wps = one_shot(
        server, model, prompt, system=system, options=options, **extra
    )
    r.ttft = ttft
    r.total_ms = total_ms
    r.wps = wps

    if not text:
        r.passed = False
        print(f"  → {RED('FAIL')} (empty response)\n")
        return

    passed = check(text)
    r.passed = passed
    status = GREEN("PASS") if passed else RED("FAIL")
    print(f"  → {status}  TTFT={ttft:.0f}ms  total={total_ms:.0f}ms  ~{wps:.1f} w/s\n")


# ── Test definitions ──────────────────────────────────────────────────────────

TASKS = [
    {
        "name": "factual",
        "prompt": "What is the capital of Japan? Answer in one word.",
        "check": lambda r: "tokyo" in r.lower(),
        "thinking": False,
    },
    {
        "name": "math",
        "prompt": "What is 17 × 23? Reply with just the number.",
        "check": lambda r: "391" in r,
        "thinking": False,
    },
    {
        "name": "code",
        "prompt": "Python one-liner to reverse a string s. Just the expression.",
        "check": lambda r: "[::-1]" in r or "reversed" in r,
        "thinking": False,
    },
    # ── Thinking-control tests (Qwen3 only) ───────────────────────────────────
    {
        "name": "think-on",
        "prompt": "What is 17 × 23? Show your reasoning, then give the answer.",
        "check": lambda r: "391" in r,
        "options": {"enable_thinking": True},
        "thinking": True,
        "note": "options.enable_thinking=true",
    },
    {
        "name": "think-off",
        "prompt": "What is 17 × 23? Reply with just the number.",
        "check": lambda r: "391" in r,
        "options": {"enable_thinking": False},
        "thinking": True,
        "note": "options.enable_thinking=false",
    },
    {
        "name": "no_think-sys",
        "prompt": "What is 17 × 23? Reply with just the number.",
        "system": "/no_think Be concise.",
        "check": lambda r: "391" in r,
        "thinking": True,
        "soft_switch": True,  # Qwen3 only — Qwen3.5/Gemma don't support /no_think
        "note": "/no_think prefix in system prompt (Qwen3 only)",
    },
    {
        "name": "tpl-kwargs",
        "prompt": "What is 17 × 23? Reply with just the number.",
        "check": lambda r: "391" in r,
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
        "thinking": True,
        "note": "chat_template_kwargs field",
    },
]


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[Result]) -> int:
    print(BOLD("─" * 80))
    print(BOLD(f"{'Model':<28} {'Task':<16} {'TTFT':>8} {'Total':>8} {'w/s':>6}  Pass"))
    print(BOLD("─" * 80))
    failures = 0
    for r in results:
        if r.skipped:
            status = YELLOW("skip")
        elif r.passed:
            status = GREEN("✓")
        else:
            status = RED("✗")
            failures += 1

        ttft  = f"{r.ttft:.0f}ms"  if r.ttft  else "—"
        total = f"{r.total_ms:.0f}ms" if r.total_ms else "—"
        wps   = f"{r.wps:.1f}"   if r.wps   else "—"
        model_short = r.model[:27]
        print(f"{model_short:<28} {r.task:<16} {ttft:>8} {total:>8} {wps:>6}  {status}")
    print(BOLD("─" * 80))
    return failures


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        metavar="MODEL",
        help="Bare tags or full HF ids (default: Qwen3-1.7B-4bit gemma-3-1b-it-bf16)",
    )
    args = parser.parse_args()

    # ── Check server reachable ────────────────────────────────────────────────
    try:
        requests.get(f"{args.server}/health", timeout=5).raise_for_status()
    except Exception:
        print(RED(f"Server not reachable at {args.server}"))
        print("Start it with:  python server.py --port 11435")
        sys.exit(1)

    results: list[Result] = []

    print(BOLD(f"\n{'═' * 80}"))
    print(BOLD("  ai-mlx-server — interactive feature benchmark"))
    print(BOLD(f"  Server: {args.server}    Models: {', '.join(args.models)}"))
    print(BOLD(f"{'═' * 80}\n"))

    for model in args.models:
        thinking_ok = _supports_thinking(model)
        print(BOLD(f"┌── {model} {'(thinking ✓)' if thinking_ok else '(no thinking)'}"))
        print()

        soft_switch_ok = _supports_soft_switch(model)
        for task in TASKS:
            needs_thinking = task.get("thinking", False)
            needs_soft_switch = task.get("soft_switch", False)
            note = task.get("note", "")
            if note:
                print(f"  {DIM(f'[{note}]')}")

            _run(
                results,
                server=args.server,
                model=model,
                task=task["name"],
                prompt=task["prompt"],
                check=task["check"],
                system=task.get("system"),
                options=task.get("options"),
                skip=(needs_thinking and not thinking_ok) or (needs_soft_switch and not soft_switch_ok),
                **(task.get("extra") or {}),
            )

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    failures = print_summary(results)
    if failures:
        print(RED(f"\n{failures} test(s) failed."))
        sys.exit(1)
    else:
        print(GREEN("\nAll tests passed."))


if __name__ == "__main__":
    main()
