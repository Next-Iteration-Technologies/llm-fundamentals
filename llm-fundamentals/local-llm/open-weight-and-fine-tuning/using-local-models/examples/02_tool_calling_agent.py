#!/usr/bin/env python3
"""A small tool-calling loop: the model may read a file in this folder.

Prerequisites
-------------
1. Install uv (https://docs.astral.sh/uv/) and Python 3.11+.
2. Install Ollama and start the server (desktop app, or `ollama serve`).
3. Check: curl http://localhost:11434/api/version
4. Pull a model that supports tools:
     ollama pull qwen3.5:0.8b
   If the 0.8B model skips the tool, use: ollama pull qwen3.5:4b
5. From the open-weight-and-fine-tuning/ folder, run: uv sync
6. Optional remote server: export OLLAMA_HOST=http://HOST:11434
7. Expected output: a log line with the tool call (read_file on README.md),
   then a short summary of this examples README.

Usage (from the open-weight-and-fine-tuning/ folder)
-----
    uv run python using-local-models/examples/02_tool_calling_agent.py
    uv run python using-local-models/examples/02_tool_calling_agent.py --model qwen3.5:4b
    uv run python using-local-models/examples/02_tool_calling_agent.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

def _add_lib() -> None:
    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        candidate = parent / "lib"
        if (candidate / "ollama_check.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError("Could not find lib/ollama_check.py")

_add_lib()
from ollama_check import ollama_base, require_ollama

from ollama import Client


DEFAULT_MODEL = "qwen3.5:0.8b"
EXAMPLES_DIR = Path(__file__).resolve().parent

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file in the examples folder. Path is a file name, not a directory traversal.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File name relative to the examples folder, for example README.md"},
            },
            "required": ["path"],
        },
    },
}


def read_file(path: str) -> str:
    """Read a file that lives inside the examples folder."""
    target = (EXAMPLES_DIR / path).resolve()
    if not str(target).startswith(str(EXAMPLES_DIR)) or not target.is_file():
        return f"error: {path!r} is not a file in the examples folder"
    if target.stat().st_size > 200_000:
        return "error: file is too large to read in this demo"
    return target.read_text(encoding="utf-8")


TOOLS = {"read_file": read_file}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Read README.md and summarise it in two bullet points.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host = ollama_base()
    if args.dry_run:
        print(f"dry-run: would connect to {host}")
        print(f"dry-run: would require model {args.model!r}")
        print(f"dry-run: would offer the read_file tool and ask: {args.prompt!r}")
        return 0

    require_ollama(args.model)
    client = Client(host=host)
    messages = [{"role": "user", "content": args.prompt}]

    for step in range(4):
        resp = client.chat(
            model=args.model,
            messages=messages,
            tools=[READ_FILE_TOOL],
            options={"temperature": 0.1, "num_ctx": 16384},
        )
        msg = resp.message
        messages.append(msg)
        if not msg.tool_calls:
            print(msg.content or "")
            return 0
        for call in msg.tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or {}
            if isinstance(raw_args, str):
                raw_args = json.loads(raw_args)
            print(f"tool call [{step}]: {name}({raw_args})")
            fn = TOOLS.get(name)
            result = fn(**raw_args) if fn else f"error: unknown tool {name}"
            messages.append({"role": "tool", "tool_name": name, "content": result})

    print("error: stopped after too many tool steps", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
