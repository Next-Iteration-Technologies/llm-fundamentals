"""Create relative MkDocs symlinks under docs/, all pointing inside this folder.

MkDocs needs a child docs_dir. The Markdown at the folder root is the source.
These links are relative, so a copy of this folder keeps them. If a zip tool
drops them, this hook recreates them on `mkdocs serve` or `mkdocs build`.
You can also run:  python ensure_docs_links.py
"""

from __future__ import annotations

from pathlib import Path

LINKS = {
    "index.md": "../index.md",
    "start.md": "../README.md",
    "assets": "../assets",
    "using-local-models": "../using-local-models",
    "fine-tuning": "../fine-tuning",
}


def ensure(root: Path | None = None) -> Path:
    root = (root or Path(__file__).resolve().parent).resolve()
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    for name, target in LINKS.items():
        dest = docs / name
        if dest.is_symlink():
            if dest.readlink().as_posix() == target and dest.exists():
                continue
            dest.unlink()
        elif dest.exists():
            continue
        dest.symlink_to(target)
    return docs


def on_startup(command, dirty):  # noqa: ARG001  (MkDocs hook)
    ensure()


if __name__ == "__main__":
    ensure()
    print("docs/ relative links are in place")
