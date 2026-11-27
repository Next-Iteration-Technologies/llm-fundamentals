#!/usr/bin/env python3
"""Answer a question about sample_policy.pdf using local embeddings and Ollama.

Prerequisites
-------------
1. Install uv (https://docs.astral.sh/uv/) and Python 3.11+.
2. Install Ollama and start the server (desktop app, or `ollama serve`).
3. Check: curl http://localhost:11434/api/version
4. Pull both models:
     ollama pull qwen3.5:0.8b
     ollama pull embeddinggemma-2:270m
5. From the open-weight-and-fine-tuning/ folder, run: uv sync
6. Optional remote server: export OLLAMA_HOST=http://HOST:11434
7. Expected output: top chunk scores, then an answer grounded in the
   Harbor Labs policy PDF. The default question is about paternity leave
   (10 working days). Pass --question to try another prompt.

Usage (from the open-weight-and-fine-tuning/ folder)
-----
    uv run python using-local-models/examples/03_local_pdf_rag.py
    uv run python using-local-models/examples/03_local_pdf_rag.py --question "Is alcohol reimbursable?"
    uv run python using-local-models/examples/03_local_pdf_rag.py --dry-run
"""
from __future__ import annotations

import argparse
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

import numpy as np
from ollama import Client
from pypdf import PdfReader


DEFAULT_CHAT = "qwen3.5:0.8b"
DEFAULT_EMBED = "embeddinggemma-2:270m"
PDF_PATH = Path(__file__).resolve().parent / "sample_policy.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_CHAT, help="Chat model tag")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED, help="Embedding model tag")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--question", default="How many days of paternity leave do employees get?")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(piece)
        if i + size >= len(text):
            break
        i += size - overlap
    return chunks


def main() -> int:
    args = parse_args()
    host = ollama_base()
    if args.dry_run:
        print(f"dry-run: would connect to {host}")
        print(f"dry-run: would require models {args.model!r} and {args.embed_model!r}")
        print(f"dry-run: would read {args.pdf} and ask {args.question!r}")
        return 0

    if not args.pdf.is_file():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    require_ollama(args.model)
    require_ollama(args.embed_model)
    client = Client(host=host)

    reader = PdfReader(str(args.pdf))
    full = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(full)
    if not chunks:
        print("error: could not extract text from the PDF", file=sys.stderr)
        return 1

    chunk_vecs = np.array(client.embed(model=args.embed_model, input=chunks).embeddings)
    q_vec = np.array(client.embed(model=args.embed_model, input=args.question).embeddings[0])
    # Embeddings from Ollama are L2-normalised; dot product is cosine similarity.
    scores = chunk_vecs @ q_vec
    top = np.argsort(scores)[-3:][::-1]
    print("=== retrieved chunks ===")
    for idx in top:
        print(f"[score {scores[idx]:.3f}] {chunks[int(idx)][:240]}")
        print("---")

    context = "\n\n---\n\n".join(chunks[int(i)] for i in top)
    reply = client.chat(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": "Answer using only the provided excerpts. If the answer is not there, say so clearly.",
            },
            {
                "role": "user",
                "content": f"Excerpts:\n{context}\n\nQuestion: {args.question}",
            },
        ],
        options={"temperature": 0.1, "num_ctx": 8192},
    )
    print("=== answer ===")
    print(reply.message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
