#!/usr/bin/env python3
"""Turn a small source tree into draft chat JSONL with path citations.

Prerequisites
-------------
1. Install uv and Python 3.11+.
2. From the open-weight-and-fine-tuning/ folder, run: uv sync.
3. No Ollama server is required.
4. Expected output: a JSONL file with one row per selected source file.

This is a starting point for a style or navigation dataset. For a live
repo assistant, retrieve current files (see rag-vs-fine-tuning.md).

Usage (from the open-weight-and-fine-tuning/ folder)
-----
    uv run python fine-tuning/scripts/02_prepare_code_repo.py --dry-run
    uv run python fine-tuning/scripts/02_prepare_code_repo.py \
        --input fine-tuning/sample_raw_docs/sample_repo --output /tmp/repo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "site", "dist", "build"}
SUFFIXES = {".py", ".ts", ".tsx", ".js", ".go", ".rs", ".java"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sample_raw_docs" / "sample_repo",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/code_repo.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.stat().st_size > 80_000:
            continue
        files.append(path)
    return files


def first_docstring(text: str) -> str:
    for quote in ('"""', "'''"):
        if text.startswith(quote):
            end = text.find(quote, 3)
            if end != -1:
                return " ".join(text[3:end].split())
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and len(stripped) > 2:
            return stripped.lstrip("# ").strip()
    return ""


def main() -> int:
    args = parse_args()
    files = iter_files(args.input) if args.input.is_dir() else []
    if args.dry_run:
        print(f"dry-run: would read {len(files)} source files from {args.input}")
        for path in files:
            try:
                rel = path.relative_to(args.input)
            except ValueError:
                rel = path
            print(f"  {rel}")
        print(f"dry-run: would write draft rows to {args.output}")
        return 0
    if not files:
        print(f"no source files in {args.input}", file=sys.stderr)
        return 1

    rows = []
    for path in files:
        rel = path.relative_to(args.input).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        doc = first_docstring(text) or f"Source file {rel}."
        rows.append({
            "messages": [
                {"role": "system", "content": "You are a repo guide. Name the file path in your answer."},
                {"role": "user", "content": f"Where is the code that deals with this: {doc}"},
                {"role": "assistant", "content": f"{doc} See `{rel}`."},
            ]
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
