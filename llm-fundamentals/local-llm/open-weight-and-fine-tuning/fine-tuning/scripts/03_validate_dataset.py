#!/usr/bin/env python3
"""Validate Harbor Labs policy JSONL: schema, footer, and basic types.

Prerequisites
-------------
1. Install uv and Python 3.11+.
2. From the open-weight-and-fine-tuning/ folder, run: uv sync.
3. No Ollama server is required.
4. Expected output: "ok: N rows in <path>" for each file, exit 0.
   On failure, each error is printed and the process exits 1.

Usage (from the open-weight-and-fine-tuning/ folder)
-----
    uv run python fine-tuning/scripts/03_validate_dataset.py --dry-run
    uv run python fine-tuning/scripts/03_validate_dataset.py fine-tuning/qwen-finetune-demo/data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FOOTER_RE = re.compile(
    r'\{\s*"policy_id"\s*:\s*(null|"[A-Z]+-\d+")\s*,\s*"escalation_needed"\s*:\s*(true|false)\s*\}\s*$',
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return [f"{path}: file is empty"]
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            errors.append(f"{path}:{i}: empty line")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{i}: invalid JSON ({exc})")
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            errors.append(f"{path}:{i}: messages must be a list with at least user and assistant")
            continue
        roles = [m.get("role") for m in messages if isinstance(m, dict)]
        if "user" not in roles or "assistant" not in roles:
            errors.append(f"{path}:{i}: missing user or assistant role")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
                errors.append(f"{path}:{i}: each message needs non-empty string content")
                break
        assistant = next((m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "assistant"), "")
        last = assistant.strip().splitlines()[-1] if assistant.strip() else ""
        match = FOOTER_RE.search(last)
        if not match:
            errors.append(f"{path}:{i}: assistant reply must end with a policy_id JSON footer")
            continue
        pid_raw, esc_raw = match.group(1), match.group(2).lower()
        if pid_raw == "null" and esc_raw == "false":
            errors.append(f"{path}:{i}: null policy_id should set escalation_needed true")
    return errors


def main() -> int:
    args = parse_args()
    paths = list(args.paths)
    if not paths:
        default = Path(__file__).resolve().parents[1] / "qwen-finetune-demo" / "data"
        paths = sorted(default.glob("*.jsonl"))
    if args.dry_run:
        print("dry-run: would validate")
        for path in paths:
            print(f"  {path}")
        return 0
    all_errors: list[str] = []
    for path in paths:
        if not path.is_file():
            all_errors.append(f"missing file: {path}")
            continue
        errs = validate_file(path)
        if errs:
            all_errors.extend(errs)
        else:
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"ok: {count} rows in {path}")
    if all_errors:
        print("\n".join(all_errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
