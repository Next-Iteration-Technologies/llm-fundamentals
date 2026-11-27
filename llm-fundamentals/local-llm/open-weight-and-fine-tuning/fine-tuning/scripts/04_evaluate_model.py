#!/usr/bin/env python3
"""Score an Ollama model on a policy JSONL eval set.

Prerequisites
-------------
1. Install uv and Python 3.11+.
2. Install Ollama and start the server (desktop app, or `ollama serve`).
3. Check: curl http://localhost:11434/api/version
4. Pull or create the model you want to score, for example:
     ollama pull qwen3.5:0.8b
   After the demo: ollama create harbor-policy -f Modelfile
5. From the open-weight-and-fine-tuning/ folder, run: uv sync
6. Optional remote server: export OLLAMA_HOST=http://HOST:11434
7. Expected output: one result line per test row, then a footer-parse
   rate and a policy_id match rate.

Usage (from the open-weight-and-fine-tuning/ folder)
-----
    uv run python fine-tuning/scripts/04_evaluate_model.py --dry-run
    uv run python fine-tuning/scripts/04_evaluate_model.py \
        --data fine-tuning/qwen-finetune-demo/data/test.jsonl --model qwen3.5:0.8b
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
        default=Path(__file__).resolve().parents[1] / "qwen-finetune-demo" / "data" / "test.jsonl",
    )
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expected_footer(assistant: str) -> dict:
    match = FOOTER_RE.search(assistant)
    if not match:
        return {}
    return json.loads(match.group(0))


def main() -> int:
    args = parse_args()
    rows = []
    if args.data.is_file():
        for line in args.data.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if args.dry_run:
        print(f"dry-run: would connect to {ollama_base()}")
        print(f"dry-run: would require model {args.model!r}")
        print(f"dry-run: would score {len(rows)} rows from {args.data}")
        return 0
    if not rows:
        print(f"no rows in {args.data}", file=sys.stderr)
        return 1

    require_ollama(args.model)
    client = Client(host=ollama_base())
    footer_ok = 0
    id_ok = 0
    for i, row in enumerate(rows, 1):
        messages = [m for m in row["messages"] if m["role"] != "assistant"]
        gold = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        expect = expected_footer(gold)
        resp = client.chat(
            model=args.model,
            messages=messages,
            options={"temperature": 0, "num_ctx": 4096},
        )
        text = resp.message.content or ""
        got = expected_footer(text)
        has_footer = bool(got)
        match_id = has_footer and got.get("policy_id") == expect.get("policy_id")
        footer_ok += int(has_footer)
        id_ok += int(match_id)
        flag = "ok" if match_id else "miss"
        print(f"[{i} {flag}] expected={expect.get('policy_id')!r} got={got.get('policy_id') if got else 'NO_FOOTER'!r}")
        if not match_id:
            print("    reply:", " ".join(text.split())[:240])

    n = len(rows)
    print(f"footer parse: {footer_ok}/{n} ({footer_ok/n:.0%})")
    print(f"policy_id match: {id_ok}/{n} ({id_ok/n:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
