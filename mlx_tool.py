#!/usr/bin/env python3
"""mlx-tool — manage and interact with local MLX models.

Usage:
    mlx-tool list                        # list downloaded models
    mlx-tool list --all                  # include non-mlx-community models
    mlx-tool list --json                 # machine-readable output
    mlx-tool delete <model>              # delete a cached model
    mlx-tool chat <model>                # interactive chat REPL

<model> can be a full repo id (mlx-community/Qwen3-4B-4bit) or a bare
name (Qwen3-4B-4bit) which auto-resolves to mlx-community/<name>.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ── Architecture sets for model-type detection ────────────────────────────────

_EMBEDDING_ARCHS = {
    "bert", "roberta", "distilbert", "electra", "xlm-roberta",
    "deberta", "albert", "mpnet", "nomic-bert", "modernbert",
    "arctic-embed", "snowflake",
}
_CHAT_ARCHS = {
    "llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3",
    "phi", "phi3", "falcon", "gpt2", "gpt_neox", "mpt", "bloom",
    "mixtral", "deepseek", "internlm", "baichuan", "chatglm",
    "starcoder", "codegen", "command-r", "olmo",
}

# ── HuggingFace cache helpers ─────────────────────────────────────────────────


def _hf_hub_dir() -> Path:
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "huggingface",
    )
    return Path(hf_home) / "hub"


def _resolve(model: str) -> str:
    """Auto-prefix bare names with mlx-community/."""
    return model if "/" in model else f"mlx-community/{model}"


def _repo_to_dir(repo: str) -> str:
    """'org/name' → 'models--org--name'."""
    return "models--" + repo.replace("/", "--")


def _repo_name(dir_name: str) -> str:
    """'models--org--name' → 'org/name'."""
    return "/".join(dir_name.removeprefix("models--").split("--"))


def _latest_snapshot(model_dir: Path) -> Path | None:
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    return max(snapshots, key=lambda p: p.stat().st_mtime) if snapshots else None


def _has_weights(snapshot_dir: Path) -> bool:
    return (
        any(snapshot_dir.glob("*.safetensors"))
        or any(snapshot_dir.glob("model*.gguf"))
        or (snapshot_dir / "weights.npz").exists()
    )


def _detect_type(snapshot_dir: Path) -> str:
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        return "unknown"
    try:
        cfg = json.loads(config_path.read_text())
        arch = cfg.get("model_type", "").lower()
        archs = [a.lower() for a in cfg.get("architectures", [])]
        all_archs = {arch} | set(archs)
        if any(a in _EMBEDDING_ARCHS or "embed" in a for a in all_archs):
            return "embedding"
        if any(a in _CHAT_ARCHS for a in all_archs):
            return "chat"
    except Exception:
        pass
    return "chat"


def _dir_size_mb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def _format_size(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


# ── list ──────────────────────────────────────────────────────────────────────


def list_local_models(mlx_community_only: bool = True) -> list[dict]:
    hub_dir = _hf_hub_dir()
    if not hub_dir.exists():
        return []
    results = []
    for entry in sorted(hub_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        repo = _repo_name(entry.name)
        if mlx_community_only and not repo.startswith("mlx-community/"):
            continue
        snapshot = _latest_snapshot(entry)
        if snapshot is None or not _has_weights(snapshot):
            continue
        size_mb = _dir_size_mb(entry)
        results.append({
            "name": repo,
            "type": _detect_type(snapshot),
            "size_mb": round(size_mb, 1),
            "path": str(snapshot),
        })
    return results


def cmd_list(args: argparse.Namespace) -> None:
    models = list_local_models(mlx_community_only=not args.all)

    if args.json:
        print(json.dumps(models, indent=2))
        return

    if not models:
        scope = "any" if args.all else "mlx-community"
        print(f"No downloaded {scope} models found in {_hf_hub_dir()}")
        if not args.all:
            print("Use --all to include models from other organisations.")
        return

    chats = [m for m in models if m["type"] == "chat"]
    embeddings = [m for m in models if m["type"] == "embedding"]
    unknowns = [m for m in models if m["type"] == "unknown"]
    col_w = max(len(m["name"]) for m in models) + 2

    def _section(title: str, items: list[dict]) -> None:
        if not items:
            return
        print(f"\n{title} ({len(items)})")
        print("─" * (col_w + 12))
        for m in items:
            print(f"  {m['name']:<{col_w}}{_format_size(m['size_mb'])}")

    print(f"Local MLX models  [{_hf_hub_dir()}]")
    _section("Chat / generation", chats)
    _section("Embedding", embeddings)
    _section("Unknown", unknowns)
    print()


# ── delete ────────────────────────────────────────────────────────────────────


def cmd_delete(args: argparse.Namespace) -> None:
    repo = _resolve(args.model)
    hub_dir = _hf_hub_dir()
    model_dir = hub_dir / _repo_to_dir(repo)

    if not model_dir.exists():
        print(f"Model not found in cache: {repo}")
        print(f"  (looked in {model_dir})")
        sys.exit(1)

    size_mb = _dir_size_mb(model_dir)
    print(f"Model : {repo}")
    print(f"Path  : {model_dir}")
    print(f"Size  : {_format_size(size_mb)}")

    if not args.yes:
        try:
            answer = input("Delete? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if answer not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    shutil.rmtree(model_dir)
    print(f"Deleted {repo}  ({_format_size(size_mb)} freed)")


# ── chat ──────────────────────────────────────────────────────────────────────


def cmd_chat(args: argparse.Namespace) -> None:
    repo = _resolve(args.model)
    print(f"Loading {repo} …")

    try:
        from mlx_lm import load, generate  # type: ignore
    except ImportError:
        print("mlx-lm is required for chat.  Run: pip install mlx-lm")
        sys.exit(1)

    model, tokenizer = load(repo)
    print(f"Ready. Type your message, or 'quit' / Ctrl-D to exit.\n")

    messages: list[dict] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                prompt = user_input
        else:
            prompt = user_input

        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )

        # Strip thinking blocks if present
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        messages.append({"role": "assistant", "content": response})
        print(f"\nAssistant: {response}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlx-tool",
        description="Manage and interact with local MLX models.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # list
    p_list = sub.add_parser("list", help="list downloaded local models")
    p_list.add_argument("--all", action="store_true", help="include non-mlx-community models")
    p_list.add_argument("--json", action="store_true", help="output as JSON")
    p_list.set_defaults(func=cmd_list)

    # delete
    p_del = sub.add_parser("delete", help="delete a cached model")
    p_del.add_argument("model", help="model name or repo id")
    p_del.add_argument("-y", "--yes", action="store_true", help="skip confirmation prompt")
    p_del.set_defaults(func=cmd_delete)

    # chat
    p_chat = sub.add_parser("chat", help="interactive chat with a model")
    p_chat.add_argument("model", help="model name or repo id")
    p_chat.add_argument("--max-tokens", type=int, default=2048, metavar="N",
                        help="max tokens per response (default: 2048)")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
