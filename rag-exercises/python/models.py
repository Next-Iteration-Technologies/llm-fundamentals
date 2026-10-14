"""Which model rag.py talks to: Nexus (OpenAI-compatible), Anthropic, Ollama, or no model at all.

Blocks come from providers.toml (fallback providers.example.toml), keys from .env or the environment.
Same shape as ai-application-architecture/llm_client.py in the earlier track, without demo and replay.

One message format for every provider; this file translates:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [{"id", "name", "arguments"}]}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "the result"}
A tool is {"name", "description", "parameters": a JSON schema}.
"""
import hashlib, json, os, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAX_ANSWER_TOKENS = 16000
MAY_READ = {"local": {"Public", "Internal", "Confidential", ""}, "tenant": {"Public", "Internal"}, "external": {"Public"}}


class SetupProblem(Exception):
    """Something on this laptop stops the call. The message says what to do."""


def load_env_file():
    """Read keys like NEXUS_API_KEY from .env in this folder. A variable already set wins."""
    env_file = HERE / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip().removeprefix("export "), value.strip().strip("'\""))


load_env_file()

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def load_providers():
    own = HERE / "providers.toml"
    with open(own if own.exists() else HERE / "providers.example.toml", "rb") as f:
        return tomllib.load(f)


PROVIDERS = load_providers()


def block(name, role):
    """The TOML block called `name`; role is "chat" or "embed" and picks the default and the error text."""
    name = name or os.environ.get(f"LLM_{role.upper()}") or PROVIDERS.get(role)
    if name not in PROVIDERS or not isinstance(PROVIDERS[name], dict):
        sys.exit(f"No {role} block called '{name}' in providers.toml (have: "
                 + ", ".join(k for k, v in PROVIDERS.items() if isinstance(v, dict)) + ").")
    return name, PROVIDERS[name]


def where(chat_name):
    """Flow 9.4: where the chat model runs. CHAT_MODEL_WHERE overrides the block."""
    w = os.environ.get("CHAT_MODEL_WHERE") or block(chat_name, "chat")[1].get("where", "external")
    if w not in MAY_READ:
        sys.exit(f"where must be local, tenant or external, not '{w}'.")
    return w


def embedder_id(embed_name):
    """What the sidecar records; a change means a full rebuild."""
    name, b = block(embed_name, "embed")
    return f"{b['kind']}:{b.get('model', name)}"


def _api_key(b):
    var = b.get("api_key_env")
    key = os.environ.get(var, "") if var else ""
    if var and not key:
        if b["kind"] == "anthropic":
            return None  # the SDK also reads ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile
        raise SetupProblem(f"{var} is not set. Put it in .env, or use --dry.")
    return key


def _timeout(b, default):
    return float(b.get("timeout") or os.environ.get("LLM_TIMEOUT") or default)


def _post(b, path, payload, timeout, label):
    import httpx
    try:
        key = _api_key(b)
        url = b["base_url"].rstrip("/") + path
        if b.get("api_version") and b["kind"] == "openai":  # Azure OpenAI style gateway: the model is in the path
            url = f"{b['base_url'].rstrip('/')}/openai/deployments/{b['model']}{path}?api-version={b['api_version']}"
        r = httpx.post(url, json=payload, timeout=_timeout(b, timeout),
                       headers={"Authorization": f"Bearer {key}"} if key else {})
        r.raise_for_status()
    except httpx.ConnectError:
        raise SetupProblem(f"Cannot reach {label} at {b['base_url']}." + (" Start it with 'ollama serve'." if b["kind"] == "ollama" else ""))
    except httpx.HTTPStatusError as e:
        raise SetupProblem(f"{label} answered {e.response.status_code}: {e.response.text.strip()[:300]}")
    return r.json()


# ------------------------------------------------------------------ chat
def chat(messages, system=None, tools=None, name=None):
    """Send a conversation (and the tools the model may call); get {"content", "tool_calls", "model"}."""
    name, b = block(name, "chat")
    kind = b["kind"]
    if kind == "anthropic":
        return _anthropic_chat(b, messages, system, tools)
    if kind == "openai":
        return _openai_chat(b, messages, system, tools, name)
    if kind == "ollama":
        return _ollama_chat(b, messages, system, tools)
    raise SetupProblem(f"Block '{name}' has kind '{kind}'; a chat block needs anthropic, openai or ollama.")


def text(reply):
    """The answer as plain text, thinking tags stripped."""
    return re.sub(r"<think>.*?</think>", "", reply["content"] or "", flags=re.S).strip()


def _anthropic_chat(b, messages, system, tools):
    import anthropic
    client = anthropic.Anthropic(api_key=_api_key(b), timeout=_timeout(b, 900))
    request = dict(model=b["model"], max_tokens=MAX_ANSWER_TOKENS, messages=anthropic_messages(messages),
                   output_config={"effort": b.get("effort", "low")},
                   betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if system:
        request["system"] = system
    if tools:
        request["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
    try:
        r = client.beta.messages.create(**request)
    except anthropic.APIConnectionError:
        raise SetupProblem("Cannot reach the Anthropic API. Check the network.")
    except anthropic.AuthenticationError:
        raise SetupProblem("The Anthropic key was refused or missing. Put ANTHROPIC_API_KEY in .env.")
    content = "(The model declined to answer this one.)" if r.stop_reason == "refusal" else \
        "".join(bl.text for bl in r.content if bl.type == "text")
    calls = [{"id": bl.id, "name": bl.name, "arguments": dict(bl.input)} for bl in r.content if bl.type == "tool_use"]
    return {"content": content, "tool_calls": calls, "model": r.model}


def _openai_chat(b, messages, system, tools, label):
    body = {"model": b["model"], "messages": ([{"role": "system", "content": system}] if system else []) + openai_messages(messages)}
    if tools:
        body["tools"] = openai_tools(tools)
    if "temperature" in b:
        body["temperature"] = b["temperature"]
    out = _post(b, "/chat/completions", body, 900, label)
    m = out["choices"][0]["message"]
    calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
             for c in m.get("tool_calls") or []]
    return {"content": m.get("content") or "", "tool_calls": calls, "model": out.get("model", b["model"])}


def _ollama_chat(b, messages, system, tools):
    body = {"model": b["model"], "messages": ([{"role": "system", "content": system}] if system else []) + ollama_messages(messages),
            "stream": False}
    if tools:
        body["tools"] = openai_tools(tools)  # Ollama takes the OpenAI tool format
    if "think" in b:
        body["think"] = b["think"]
    out = _post(b, "/api/chat", body, 900, "Ollama")
    m = out["message"]
    calls = [{"id": f"call_{n}", "name": c["function"]["name"], "arguments": dict(c["function"].get("arguments") or {})}
             for n, c in enumerate(m.get("tool_calls") or [], 1)]
    return {"content": m.get("content", ""), "tool_calls": calls, "model": b["model"]}


# ------------------------------------------------------------------ embeddings
def embed(texts, kind, name=None):
    """Vectors for texts; kind is "query" or "doc" (some models want to know which)."""
    name, b = block(name, "embed")
    if b["kind"] == "hash":
        return hash_embed(texts)
    if b.get("model", "").startswith("nomic"):  # nomic-embed-text task prefixes
        texts = [("search_query: " if kind == "query" else "search_document: ") + t for t in texts]
    vecs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        if b["kind"] == "openai":
            out = _post(b, "/embeddings", {"model": b["model"], "input": batch}, 120, name)
            vecs += [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]
        elif b["kind"] == "ollama":
            vecs += _post(b, "/api/embed", {"model": b["model"], "input": batch}, 300, "Ollama")["embeddings"]
        else:
            raise SetupProblem(f"Block '{name}' has kind '{b['kind']}'; an embed block needs openai, ollama or hash. "
                               "Anthropic has no embeddings endpoint: pair --chat anthropic with another embed block.")
    return np.array(vecs, dtype=np.float32)


def hash_embed(texts, dim=768):
    """No model: feature hashing of words and character trigrams. Lexical, not semantic."""
    tok = lambda s: re.findall(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", s.lower())
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for r, t in enumerate(texts):
        for w in tok(t) + [w[i:i + 3] for w in tok(t) for i in range(len(w) - 2)]:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            out[r, h % dim] += 1 if (h >> 64) & 1 else -1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def reachable(name, role):
    """(ok, advice) for `python rag.py check`."""
    name, b = block(name, role)
    kind = b["kind"]
    try:
        if kind == "hash":
            return True, "no model needed"
        if kind == "anthropic":
            import anthropic
            anthropic.Anthropic(api_key=_api_key(b)).models.retrieve(b["model"])
        elif kind == "openai":  # a gateway may not serve /models: make one tiny real call instead
            if role == "embed":
                embed(["ping"], "query", name)
            else:
                _post(b, "/chat/completions", {"model": b["model"], "messages": [{"role": "user", "content": "Say ok."}],
                                                "max_tokens": 5}, 60, name)
        elif kind == "ollama":
            import httpx
            tags = httpx.get(b["base_url"].rstrip("/") + "/api/tags", timeout=5).json()
            names = {m["name"] for m in tags.get("models", [])} | {m["name"].split(":")[0] for m in tags.get("models", [])}
            if b["model"] not in names:
                return False, f"run: ollama pull {b['model']}"
        return True, b.get("model", "")
    except SetupProblem as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


# ------------------------------------------------------------------ one message format, translated for each API
def anthropic_messages(messages):
    """Tool calls become tool_use blocks; tool results go back inside a user message."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            blocks = [{"type": "text", "text": m["content"]}] if m["content"] else []
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]} for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            result = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(result)  # several results from one turn share a message
            else:
                out.append({"role": "user", "content": [result]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def openai_messages(messages):
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m["content"] or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def ollama_messages(messages):
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m["content"], "tool_calls": [
                {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def openai_tools(tools):
    return [{"type": "function", "function": t} for t in tools]
