#!/usr/bin/env python3
"""Turn a folder of policy Markdown or text files into a draft SFT JSONL.

Prerequisites
-------------
1. Install uv and Python 3.11+.
2. From the open-weight-and-fine-tuning/ folder, run: uv sync.
3. No Ollama server is required.
4. Expected output: a JSONL file with one summary pair and one refusal
   stub per input document, plus a printed row count.

Usage (from the open-weight-and-fine-tuning/ folder)
-----
    uv run python fine-tuning/scripts/01_prepare_enterprise_docs.py --dry-run
    uv run python fine-tuning/scripts/01_prepare_enterprise_docs.py \
        --input fine-tuning/sample_raw_docs --output /tmp/enterprise.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SYSTEM = (
    "You are Harbor Labs' internal policy assistant. Answer only from "
    "Harbor Labs policy. If the policy set does not contain the answer, "
    "say you do not have it and set escalation_needed to true. End every "
    "reply with a JSON object on its own last line of the form "
    '{"policy_id": "HR-14" or null, "escalation_needed": true or false}.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sample_raw_docs",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/enterprise_docs.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def policy_id_from(name: str, text: str) -> str | None:
    stem = Path(name).stem.upper()
    match = re.search(r"([A-Z]+-\d+)", stem) or re.search(r"([A-Z]+-\d+)", text)
    return match.group(1) if match else None


def iter_docs(folder: Path) -> list[Path]:
    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"} and "sample_repo" not in path.parts:
            files.append(path)
    return files


def main() -> int:
    args = parse_args()
    files = iter_docs(args.input) if args.input.is_dir() else []
    if args.dry_run:
        print(f"dry-run: would read {len(files)} docs from {args.input}")
        for path in files:
            print(f"  {path.name}")
        print(f"dry-run: would write draft rows to {args.output}")
        return 0
    if not files:
        print(f"no .md or .txt files in {args.input}", file=sys.stderr)
        return 1

    rows = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        pid = policy_id_from(path.name, text) or "UNKNOWN"
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        summary = " ".join(text.split())
        if len(summary) > 500:
            summary = summary[:497] + "..."
        footer = json.dumps({"policy_id": pid, "escalation_needed": False})
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"What does policy {pid} ({title}) say? Summarise it."},
                {"role": "assistant", "content": summary + "\n\n" + footer},
            ]
        })
        refusal = json.dumps({"policy_id": None, "escalation_needed": True})
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Does {pid} mention a monthly wellness stipend?"},
                {"role": "assistant", "content": "I do not have a wellness stipend in that policy.\n\n" + refusal},
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
