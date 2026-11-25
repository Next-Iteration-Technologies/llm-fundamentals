#!/usr/bin/env python3
"""Ask a base Ollama model and a tuned model the same test questions.

Prerequisites
-------------
1. Install uv and Python 3.11+.
2. Install Ollama and start the server (desktop app, or `ollama serve`).
3. Check: curl http://localhost:11434/api/version
4. Pull the base model, and create the tuned model from the Modelfile:
     ollama pull qwen3.5:0.8b
     ollama create harbor-policy -f Modelfile
5. From the open-weight-and-fine-tuning/ folder, run: uv sync
6. Optional remote server: export OLLAMA_HOST=http://HOST:11434
7. Expected output: for each test row, both replies, then footer and
   policy_id match rates for each model.

Usage (from the open-weight-and-fine-tuning/ folder)
-----
    uv run python fine-tuning/qwen-finetune-demo/compare_in_ollama.py --dry-run
    uv run python fine-tuning/qwen-finetune-demo/compare_in_ollama.py \
        --base qwen3.5:0.8b --tuned harbor-policy
"""
from __future__ import annotations

import argparse
import json
import re
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
from ollama_check import ollama_base, require_ollama  # noqa: E402
from ollama import Client  # noqa: E402

FOOTER_RE = re.compile(
    r'\{\s*"policy_id"\s*:\s*(null|"[A-Z]+-\d+")\s*,\s*"escalation_needed"\s*:\s*(true|false)\s*\}',
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "test.jsonl",
    )
    parser.add_argument("--base", default="qwen3.5:0.8b")
    parser.add_argument("--tuned", default="harbor-policy")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def footer(text: str) -> dict:
    match = FOOTER_RE.search(text or "")
    return json.loads(match.group(0)) if match else {}


def ask(client: Client, model: str, messages: list[dict]) -> str:
    resp = client.chat(model=model, messages=messages, options={"temperature": 0, "num_ctx": 4096})
    return resp.message.content or ""


def main() -> int:
    args = parse_args()
    rows = []
    if args.data.is_file():
        rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.dry_run:
        print(f"dry-run: would connect to {ollama_base()}")
        print(f"dry-run: would require models {args.base!r} and {args.tuned!r}")
        print(f"dry-run: would compare {len(rows)} rows from {args.data}")
        return 0
    if not rows:
        print(f"no rows in {args.data}", file=sys.stderr)
        return 1

    require_ollama(args.base)
    require_ollama(args.tuned)
    client = Client(host=ollama_base())
    stats = {args.base: {"footer": 0, "id": 0}, args.tuned: {"footer": 0, "id": 0}}

    for i, row in enumerate(rows, 1):
        messages = [m for m in row["messages"] if m["role"] != "assistant"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        gold = footer(next(m["content"] for m in row["messages"] if m["role"] == "assistant"))
        print(f"\n== {i}. {user}")
        print(f"expected policy_id={gold.get('policy_id')!r}")
        for name in (args.base, args.tuned):
            text = ask(client, name, messages)
            got = footer(text)
            stats[name]["footer"] += int(bool(got))
            stats[name]["id"] += int(bool(got) and got.get("policy_id") == gold.get("policy_id"))
            label = "FOOTER" if got else "NO_FOOTER"
            print(f"-- {name} [{label} policy_id={got.get('policy_id') if got else None!r}]")
            print(text.strip()[:500])

    n = len(rows)
    print("\n=== summary ===")
    for name, vals in stats.items():
        print(f"{name}: footer {vals['footer']}/{n} ({vals['footer']/n:.0%}), "
              f"policy_id {vals['id']}/{n} ({vals['id']/n:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
