#!/usr/bin/env python3
"""Send a chat request and a structured JSON request to a local Ollama server.

Prerequisites
-------------
1. Install uv (https://docs.astral.sh/uv/) and Python 3.11+.
2. Install Ollama (https://ollama.com) and start the server
   (desktop app, or `ollama serve`).
3. Check the server is up:
     curl http://localhost:11434/api/version
   You should see a JSON body with a "version" field.
4. Pull the model:
     ollama pull qwen3.5:0.8b
   Optional stronger model: ollama pull qwen3.5:4b
5. From the open-weight-and-fine-tuning/ folder, run: uv sync
6. Optional remote server: export OLLAMA_HOST=http://HOST:11434
7. Expected output: a one-sentence greeting, then a JSON object
   with "city" and "status" keys.

Usage (from the open-weight-and-fine-tuning/ folder)
-----
    uv run python using-local-models/examples/01_chat_api.py
    uv run python using-local-models/examples/01_chat_api.py --model qwen3.5:4b
    uv run python using-local-models/examples/01_chat_api.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned calls and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host = ollama_base()
    if args.dry_run:
        print(f"dry-run: would connect to {host}")
        print(f"dry-run: would require model {args.model!r}")
        print("dry-run: would send a greeting chat, then a JSON-format chat")
        return 0

    require_ollama(args.model)
    client = Client(host=host)

    greeting = client.chat(
        model=args.model,
        messages=[{"role": "user", "content": "Say hello in one short sentence, as a colleague starting a workshop."}],
        options={"temperature": 0.3},
    )
    print("=== chat ===")
    print(greeting.message.content)
    print()

    structured = client.chat(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": "Reply with JSON only, no markdown, with keys city and status. city is Mumbai, status is ok.",
            }
        ],
        format="json",
        options={"temperature": 0},
    )
    print("=== structured ===")
    raw = structured.message.content or ""
    print(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("warning: reply was not valid JSON", file=sys.stderr)
        return 1
    if "city" not in data or "status" not in data:
        print("warning: JSON is missing city or status", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
