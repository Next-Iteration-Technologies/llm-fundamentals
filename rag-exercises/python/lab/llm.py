"""Which chat model answers and which one judges, and every reply saved so a rerun costs nothing.

Answer model: --chat, else LLM_CHAT, else `chat` in providers.toml.
Judge:        --judge, else LLM_JUDGE, else `judge` in providers.toml, else the answer model.
"""
import os, time

import models

from . import cache


def chat_name(name=None):
    return models.block(name, "chat")[0]


def judge_name(name=None):
    j = name or os.environ.get("LLM_JUDGE") or (models.PROVIDERS.get("judge") if isinstance(models.PROVIDERS.get("judge"), str) else None)
    return models.block(j, "chat")[0] if j else chat_name()


def describe(name):
    n, b = models.block(name, "chat")
    return f"{n} ({b.get('model', b['kind'])})"


def ask(name, system, prompt, fresh=False, cache_system=False):
    """One chat call: {text, model, seconds, usage, saved}. saved is the date of a reused reply, else None.
    Raises models.SetupProblem (and PromptTooBig) for the caller to report."""
    n, b = models.block(name, "chat")
    k = cache.key("chat", n, b.get("kind"), b.get("model"), system or "", prompt)
    if not fresh:
        hit, saved = cache.get(k)
        if hit:
            return dict(hit, saved=saved)
    t0 = time.time()
    r = models.chat([{"role": "user", "content": prompt}], system=system, name=n, cache_system=cache_system)
    out = {"text": models.text(r), "model": r.get("model", b.get("model")), "seconds": round(time.time() - t0, 1),
           "usage": r.get("usage") or {}}
    cache.put(k, out)
    return dict(out, saved=None)
