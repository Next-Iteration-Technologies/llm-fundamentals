"""Shared health checks for a local or remote Ollama server.

Scripts in this workshop import require_ollama() so they fail early with a
clear message when the server is down or the model has not been pulled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

DEFAULT_HOST = "http://localhost:11434"


def ollama_base() -> str:
    """Return the server base URL, honouring OLLAMA_HOST."""
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def check_server(timeout: float = 3.0) -> tuple[bool, str]:
    url = ollama_base() + "/api/version"
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        version = response.json().get("version", "unknown")
        return True, str(version)
    except httpx.ConnectError:
        return False, (
            f"Cannot reach the Ollama server at {ollama_base()}.\n"
            "Start the desktop app, or run `ollama serve` in a terminal.\n"
            "Then check with: curl http://localhost:11434/api/version\n"
            "If the server is on another machine, set OLLAMA_HOST, for example:\n"
            "  export OLLAMA_HOST=http://192.168.1.20:11434"
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama at {ollama_base()} returned an error: {exc}"


def list_models(timeout: float = 5.0) -> list[str]:
    url = ollama_base() + "/api/tags"
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return [m.get("name", "") for m in response.json().get("models", [])]


def model_present(model: str, names: list[str]) -> bool:
    wanted = model.lower()
    wanted_base = wanted.split(":")[0]
    wanted_tag = wanted.split(":")[1] if ":" in wanted else "latest"
    for name in names:
        n = name.lower()
        base, _, tag = n.partition(":")
        tag = tag or "latest"
        if n == wanted or n.startswith(wanted):
            return True
        if base == wanted_base and (tag == wanted_tag or wanted_tag == "latest"):
            return True
        if wanted == wanted_base and base == wanted_base:
            return True
    return False


def require_ollama(model: str | None = None) -> str:
    """Exit with a friendly message if the server or model is missing.

    Returns the detected server version when the check passes.
    """
    ok, msg = check_server()
    if not ok:
        print(msg, file=sys.stderr)
        raise SystemExit(2)
    version = msg
    if model:
        try:
            names = list_models()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not list models at {ollama_base()}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        if not model_present(model, names):
            available = ", ".join(names) if names else "(none)"
            print(
                f"Model {model!r} is not on this Ollama server.\n"
                f"Pull it with: ollama pull {model}\n"
                "If you use a remote server, run the pull on that machine, "
                "or set OLLAMA_HOST first.\n"
                f"Models currently available: {available}",
                file=sys.stderr,
            )
            raise SystemExit(3)
    return version


def add_lib_to_path(start: Path | None = None) -> Path:
    """Insert the workshop lib/ directory on sys.path and return it."""
    here = start or Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / "lib"
        if (candidate / "ollama_check.py").exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return candidate
    raise FileNotFoundError("Could not find lib/ollama_check.py above this file")
