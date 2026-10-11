"""Set up the Contoso RAG lab, or switch the model it uses. Same commands on Windows and macOS.

  python setup.py                    first time: virtual environment, packages, config files, pick a provider, check
  python setup.py nexus              the same, with the provider chosen up front
  python setup.py provider           switch provider later (asks which)
  python setup.py provider ollama    switch without asking

Providers:
  nexus      the company gateway (the room's default); needs NEXUS_API_KEY
  anthropic  Claude; needs ANTHROPIC_API_KEY; Anthropic has no embeddings, so Ollama or the hashed stand-in embeds
  ollama     a local model; needs Ollama running, no key
  offline    no model at all; run every command with --dry

On macOS, type python3 where this says python. Only the standard library is used here, so it runs before any install.
"""
import getpass, json, os, re, shutil, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TOML, ENV = HERE / "providers.toml", HERE / ".env"
OLLAMA = "http://localhost:11434"

PROVIDERS = {  # name: (what it is, chat block, embed block, key variable)
    "nexus": ("the company gateway (the room's default)", "nexus", "nexus-embed", "NEXUS_API_KEY"),
    "anthropic": ("Claude, with an Anthropic API key", "anthropic", None, "ANTHROPIC_API_KEY"),
    "ollama": ("a local model in Ollama, no key", "ollama", "ollama-embed", None),
    "offline": ("no model: retrieval only, every command with --dry", None, "hash", None),
}


def say(text=""):
    print(text, flush=True)


def ask(question, default=""):
    answer = input(f"{question} [{default}]: " if default else f"{question}: ").strip()
    return answer or default


# ---------- install ----------

def install():
    if sys.version_info < (3, 10):
        sys.exit(f"Python {sys.version.split()[0]} is too old: install Python 3.10, 3.11 or 3.12 and run this again.")
    if not VENV_PY.exists():
        say("1/3  Creating the virtual environment in .venv ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    else:
        say("1/3  The virtual environment .venv is already there.")
    say("2/3  Installing the packages. The first time this takes a few minutes (the reranker pulls in torch) ...")
    subprocess.run([str(VENV_PY), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                    "-r", str(HERE / "requirements.txt")], check=True)
    say("3/3  Config files ...")
    for own, example in ((TOML, "providers.example.toml"), (ENV, ".env.example")):
        if not own.exists():
            shutil.copy(HERE / example, own)
            say(f"     created {own.name} from {example}")


# ---------- the two config files ----------

def read(path, example):
    if not path.exists():
        shutil.copy(HERE / example, path)
    return path.read_text(encoding="utf-8")


def set_default(text, role, block):
    """Point the top-level `chat = "..."` or `embed = "..."` line at a block, keeping its comment."""
    return re.sub(rf'^({role}\s*=\s*)"[^"]*"', rf'\g<1>"{block}"', text, count=1, flags=re.M)


def current_default(text, role):
    m = re.search(rf'^{role}\s*=\s*"([^"]*)"', text, flags=re.M)
    return m.group(1) if m else None


def get_in_block(text, block, key):
    section = re.search(rf"^\[{re.escape(block)}\][^\n]*\n(.*?)(?=^\[|\Z)", text, flags=re.M | re.S)
    m = section and re.search(rf'^{key}\s*=\s*"([^"]*)"', section.group(1), flags=re.M)
    return m.group(1) if m else ""


def set_in_block(text, block, key, value):
    def fix(section):
        return re.sub(rf'^({key}\s*=\s*)"[^"]*"', lambda m: f'{m.group(1)}"{value}"', section.group(0), count=1, flags=re.M)
    return re.sub(rf"^\[{re.escape(block)}\][^\n]*\n.*?(?=^\[|\Z)", fix, text, count=1, flags=re.M | re.S)


def env_value(name):
    if os.environ.get(name):
        return os.environ[name]
    m = re.search(rf"^{name}=(.*)$", read(ENV, ".env.example"), flags=re.M)
    return m.group(1).strip().strip("'\"") if m else ""


def set_env_value(name, value):
    text = read(ENV, ".env.example")
    if re.search(rf"^{name}=", text, flags=re.M):
        text = re.sub(rf"^{name}=.*$", lambda m: f"{name}={value}", text, count=1, flags=re.M)
    else:
        text = text.rstrip("\n") + f"\n{name}={value}\n"
    ENV.write_text(text, encoding="utf-8")


# ---------- choosing ----------

def ollama_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            return [m["name"].split(":latest")[0] for m in json.load(r)["models"]]
    except Exception:
        return None


def pick_provider():
    say("\nWhich model should the lab use?")
    names = list(PROVIDERS)
    for i, name in enumerate(names, 1):
        say(f"  {i}. {name:<10} {PROVIDERS[name][0]}")
    while True:
        answer = ask("Number or name", "1").lower()
        if answer in PROVIDERS:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        say("  Type 1 to 4, or a name from the list.")


def ask_key(var):
    have = env_value(var)
    if have:
        new = getpass.getpass(f"{var} is set ({have[:10]}...). Paste a new one, or press Enter to keep it: ").strip()
    else:
        new = getpass.getpass(f"Paste your {var} (the text stays hidden): ").strip()
    key = new or have
    if new:
        set_env_value(var, new)
        say(f"  saved {var} in .env")
    if var == "ANTHROPIC_API_KEY" and key and not key.startswith("sk-ant-api"):
        say("  Warning: Anthropic API keys start with sk-ant-api. Make one at console.anthropic.com > API keys.")
    if not key:
        say(f"  No key yet: the check below will fail until {var} is in .env.")


def ask_nexus(text):
    say("\nNexus details (ask the trainer). Press Enter to keep what is shown.")
    base = ask("  Base URL", get_in_block(text, "nexus", "base_url"))
    chat_model = ask("  Chat model", get_in_block(text, "nexus", "model"))
    embed_model = ask("  Embedding model", get_in_block(text, "nexus-embed", "model"))
    text = set_in_block(text, "nexus", "base_url", base)
    text = set_in_block(text, "nexus", "model", chat_model)
    text = set_in_block(text, "nexus-embed", "base_url", base)
    return set_in_block(text, "nexus-embed", "model", embed_model)


def choose(name=None):
    name = name or pick_provider()
    if name not in PROVIDERS:
        sys.exit(f"Unknown provider '{name}'. Choose one of: {', '.join(PROVIDERS)}.")
    _, chat, embed, key_var = PROVIDERS[name]
    text = read(TOML, "providers.example.toml")
    old_embed = current_default(text, "embed")

    if name == "nexus":
        text = ask_nexus(text)
    if name in ("anthropic", "ollama"):
        have = ollama_models()
        if have is None:
            say(f"\nOllama doesn't answer at {OLLAMA}.")
        if name == "ollama":
            for block in ("ollama", "ollama-embed"):
                model = get_in_block(text, block, "model")
                if have is not None and model not in have:
                    say(f"  Ollama hasn't got {model} yet. Run:  ollama pull {model}")
        else:  # anthropic: no embeddings endpoint
            embed = "ollama-embed" if have and get_in_block(text, "ollama-embed", "model") in have else "hash"
            if embed == "hash":
                say("  Anthropic has no embeddings. Without Ollama the hashed stand-in embeds: lexical only,\n"
                    "  so the semantic side of E2 gets weaker. With Ollama: ollama pull nomic-embed-text, then run this again.")
    if key_var:
        say("")
        ask_key(key_var)

    if chat:
        text = set_default(text, "chat", chat)
    text = set_default(text, "embed", embed)
    TOML.write_text(text, encoding="utf-8")
    say(f"\nproviders.toml now uses chat = {chat or current_default(text, 'chat')}, embed = {embed}.")
    if old_embed and old_embed != embed and (HERE / ".store").exists():
        say("The embed block changed: an index built with the old one doesn't fit. Run:  python rag.py ingest")
    return name


def check(name):
    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    say("\nChecking ...")
    if name == "offline":
        say("Offline: no model to check. Add --dry to every command, for example:\n"
            '  python rag.py ingest --dry\n  python rag.py ask "why did we drop vendor A?" --dry')
        return
    subprocess.run([str(py), str(HERE / "rag.py"), "check"])
    say('\nNext:  python rag.py ingest   then   python rag.py ask "why did we drop vendor A?"')
    say("Switch models any time with:  python setup.py provider")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] in (["-h"], ["--help"]):
        say(__doc__)
    elif args[:1] == ["provider"]:
        check(choose(args[1] if len(args) > 1 else None))
    elif len(args) <= 1:
        install()
        check(choose(args[0] if args else None))
    else:
        sys.exit(__doc__)
