"""
Shared helpers for the multi-agent-pattern labs.

Every lab talks to the same model: Nexus, read entirely from the environment
(NEXUS_BASE_URL, NEXUS_API_KEY, optionally NEXUS_MODEL). Nothing is hardcoded,
so each laptop in the room can point at its own Nexus endpoint via .env.

Labs also reuse the mock DAS archive (archive.py, one level up) so "find
invoice 1234" is a real lookup against data/documents.json.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root, for archive.py

import httpx

import archive


def _load_env_file() -> None:
    """Read NEXUS_* from .env in the project root. A variable already set wins."""
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


_load_env_file()

_REQUIRED = ("NEXUS_BASE_URL", "NEXUS_API_KEY")
_MISSING = [name for name in _REQUIRED if not os.environ.get(name)]
if _MISSING:
    sys.exit(
        f"Missing {', '.join(_MISSING)}. Set them in .env (see .env.example) "
        "or export them before running a lab."
    )

NEXUS_BASE_URL = os.environ["NEXUS_BASE_URL"]
NEXUS_API_KEY = os.environ["NEXUS_API_KEY"]
NEXUS_MODEL = os.environ.get("NEXUS_MODEL", "nexus-default")


def ask(role_prompt: str, user_text: str) -> str:
    """One Nexus chat call: a role's system prompt plus the text it's asked about."""
    response = httpx.post(
        NEXUS_BASE_URL.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {NEXUS_API_KEY}"},
        json={
            "model": NEXUS_MODEL,
            "messages": [
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": user_text},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def lookup_invoice(doc_id: str) -> dict | None:
    """The DAS archive lookup every lab's scenario starts from."""
    return archive.find_document(doc_id)


def step(label: str, detail: str = "") -> None:
    """Print one trace line, so the pattern's flow is visible while a lab runs."""
    print(f"[{label}] {detail}" if detail else f"[{label}]")
