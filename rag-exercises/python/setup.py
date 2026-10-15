"""Set up the Contoso RAG lab, or switch the models it uses. Same commands on Windows and macOS.

  python setup.py                          first time: virtual environment, packages, config files, pick the models, check
  python setup.py nexus                    the same, with the chat provider chosen up front (students)
  python setup.py anthropic ollama-embed   the same, with the chat provider and the embedding chosen up front (trainer)
  python setup.py provider                 switch later (asks which)
  python setup.py provider nexus bge-small switch without asking

Chat providers (they answer, and judge the answers in LAB chapters 7 and 9):
  nexus      the company gateway (students); needs NEXUS_API_KEY
  anthropic  Claude: Sonnet answers, Opus judges; needs ANTHROPIC_API_KEY
  ollama     a local model in Ollama, no key; only where Ollama is allowed
  offline    no model at all; run every command with --dry

Embeddings (what the chunks and questions are turned into); only the ones this laptop can use are offered:
  ollama-embed   nomic-embed-text in Ollama, where Ollama is allowed
  nexus-embed    the gateway's embedding model, if Nexus serves one
  bge-small      inside Python, ~130 MB from Hugging Face once; works everywhere
  hash           no model: a lexical stand-in

On macOS, type python3 where this says python. Only the standard library is used here, so it runs before any install.
"""
import getpass, json, os, re, shutil, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TOML, ENV, EXAMPLE = HERE / "providers.toml", HERE / ".env", HERE / "providers.example.toml"
OLLAMA = "http://localhost:11434"

PROVIDERS = {  # name: (what it is, chat block, judge block, key variable)
    "nexus": ("the company gateway (students)", "nexus", "nexus", "NEXUS_API_KEY"),
    "anthropic": ("Claude: Sonnet answers, Opus judges (the trainer)", "sonnet", "opus", "ANTHROPIC_API_KEY"),
    "ollama": ("a local model in Ollama, no key", "ollama", "ollama", None),
    "offline": ("no model: retrieval only, every command with --dry", None, None, None),
}
EMBEDS = {  # in order of preference: the first one this laptop can use is the default
    "ollama-embed": "nomic-embed-text in Ollama, on this laptop",
    "nexus-embed": "the gateway's embedding model",
    "bge-small": "inside Python, ~130 MB from Hugging Face once",
    "hash": "no model: a lexical stand-in, weak on meaning",
}
HF_HINT = ("  Hugging Face doesn't answer from here. Set HF_ENDPOINT to an internal mirror, or copy the folder\n"
           "  .cache/huggingface from a laptop that has the models into your home folder, then run this again.")


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
    for own, example in ((TOML, EXAMPLE.name), (ENV, ".env.example")):
        if not own.exists():
            shutil.copy(HERE / example, own)
            say(f"     created {own.name} from {example}")


# ---------- the two config files ----------

def read(path, example):
    if not path.exists():
        shutil.copy(HERE / example, path)
    return path.read_text(encoding="utf-8")


def set_default(text, role, block):
    """Point the top-level `chat = "..."`, `judge = "..."` or `embed = "..."` line at a block, keeping its comment.
    A missing line goes in under `chat =`."""
    if re.search(rf"^{role}\s*=", text, flags=re.M):
        return re.sub(rf'^({role}\s*=\s*)"[^"]*"', lambda m: f'{m.group(1)}"{block}"', text, count=1, flags=re.M)
    return re.sub(r"^(chat\s*=[^\n]*\n)", lambda m: f'{m.group(1)}{role} = "{block}"\n', text, count=1, flags=re.M)


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


def add_missing(text):
    """An older providers.toml lacks blocks the lab now uses (sonnet, opus, bge-small, ...): append them from the
    example, and the top-level judge line. What is already there is kept as it is."""
    example = EXAMPLE.read_text(encoding="utf-8")
    have = set(re.findall(r"^\[([^\]]+)\]", text, flags=re.M))
    added = []
    for name in re.findall(r"^\[([^\]]+)\]", example, flags=re.M):
        if name not in have:
            section = re.search(rf"^\[{re.escape(name)}\][^\n]*\n(?:[^\n]+\n?)*", example, flags=re.M).group(0)
            text = text.rstrip("\n") + "\n\n" + section.rstrip("\n") + "\n"
            added.append(name)
    judge = re.search(r"^judge\s*=[^\n]*", example, flags=re.M)
    if judge and not re.search(r"^judge\s*=", text, flags=re.M):
        text = re.sub(r"^(chat\s*=[^\n]*\n)", lambda m: m.group(1) + judge.group(0) + "\n", text, count=1, flags=re.M)
        added.append("judge")
    return text, added


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


# ---------- Ollama (optional: it may not be allowed on every laptop) ----------

def ollama_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            return [m["name"].split(":latest")[0] for m in json.load(r)["models"]]
    except Exception:
        return None


def ollama_exe():
    """On PATH, or where the Windows installer puts it (PATH reaches new windows only)."""
    found = shutil.which("ollama")
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return found or (str(local) if os.name == "nt" and local.exists() else None)


def pull(model):
    """Offer `ollama pull MODEL`; True when the model is there afterwards."""
    if ask(f"  Ollama hasn't got {model} yet. Pull it now? (y/n)", "y").lower() != "y":
        return False
    exe = ollama_exe()
    if not exe:
        say(f"  ollama isn't on the PATH of this window. Run:  ollama pull {model}   then run this again.")
        return False
    return subprocess.run([exe, "pull", model]).returncode == 0


# ---------- choosing ----------

def pick_provider():
    say("\nWhich model should answer (and judge the answers)?")
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
    if var == "ANTHROPIC_API_KEY" and key and not key.startswith("sk-ant-"):
        say("  Warning: Anthropic keys start with sk-ant-. Make one at console.anthropic.com > API keys.")
    if not key:
        say(f"  No key yet: the check below will fail until {var} is in .env.")


def nexus_models(base, key):
    """The model names the gateway lists (GET {base}/models, OpenAI format), or None if it doesn't serve that."""
    try:
        req = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return sorted(m["id"] for m in json.load(r).get("data", [])) or None
    except Exception:
        return None


def pick_model(label, options, current, none=False):
    """Pick one of the gateway's models by number; a typed name works too (a gateway may not list them all).
    none=True adds 0: no such model, returned as "-"."""
    say(f"\n  {label}:")
    if none:
        say("    0. none")
    for i, m in enumerate(options, 1):
        say(f"    {i}. {m}")
    default = str(options.index(current) + 1) if current in options else ("1" if options else "0")
    while True:
        answer = ask("  Number or name", default)
        if none and answer == "0":
            return "-"
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer and not answer.isdigit():
            return answer
        say(f"  Type {'0' if none else '1'} to {len(options)}, or a model name.")


def ask_nexus(text):
    say("\nNexus details (ask the trainer). Press Enter to keep what is shown.")
    base = ask("  Base URL", get_in_block(text, "nexus", "base_url"))
    chat_now, embed_now = get_in_block(text, "nexus", "model"), get_in_block(text, "nexus-embed", "model")
    models = nexus_models(base, env_value("NEXUS_API_KEY"))
    if models:
        embeds = [m for m in models if "embed" in m.lower()]
        chats = [m for m in models if m not in embeds] or models
        say(f"  Nexus lists {len(models)} models.")
        chat_model = pick_model("Chat model (it answers, and judges the answers)", chats, chat_now)
        if not embeds:
            say("\n  None of them looks like an embedding model: pick 0 unless you know one.")
        embed_model = pick_model("Embedding model", embeds or models, embed_now if embeds else None, none=True)
    else:
        say("  Nexus doesn't list its models here (GET /models): type the names.")
        chat_model = ask("  Chat model", chat_now)
        embed_model = ask("  Embedding model (type - if Nexus has none)", embed_now)
    text = set_in_block(text, "nexus", "base_url", base)
    text = set_in_block(text, "nexus", "model", chat_model)
    text = set_in_block(text, "nexus-embed", "base_url", base)
    return set_in_block(text, "nexus-embed", "model", embed_model)


def embed_options(name, text, given=None):
    """{embed block: None if this laptop can use it, else the reason it can't}."""
    if name == "offline":
        return {"hash": None}
    out = {}
    have = ollama_models()
    model = get_in_block(text, "ollama-embed", "model")
    if have is None:
        out["ollama-embed"] = f"Ollama doesn't answer at {OLLAMA}"
    elif model in have or (given in (None, "ollama-embed") and pull(model)):
        out["ollama-embed"] = None
    else:
        out["ollama-embed"] = f"Ollama hasn't got {model} (ollama pull {model})"
    nexus_model = get_in_block(text, "nexus-embed", "model")
    if name != "nexus":
        out["nexus-embed"] = "only with the nexus provider"
    elif nexus_model in ("", "-", "EMBED-MODEL-NAME"):
        out["nexus-embed"] = "no Nexus embedding model given"
    else:
        out["nexus-embed"] = None
    out["bge-small"] = out["hash"] = None
    first = ["nexus-embed"] if name == "nexus" else []  # the gateway's own model is the students' default
    return {e: out[e] for e in first + [e for e in out if e not in first]}


def pick_embed(name, text, given=None):
    options = embed_options(name, text, given)
    usable = [e for e, why in options.items() if why is None]
    if given:
        if given not in EMBEDS:
            sys.exit(f"Unknown embedding '{given}'. Choose one of: {', '.join(EMBEDS)}.")
        if options.get(given, "not with this provider") is not None:
            sys.exit(f"Can't use {given} here: {options.get(given, 'not with this provider')}.")
        return given
    if len(usable) == 1:
        return usable[0]
    say("\nWhich embedding model? (Chapter 5 of LAB.md compares them all; this one is the default.)")
    for i, e in enumerate(usable, 1):
        say(f"  {i}. {e:<13} {EMBEDS[e]}")
    for e, why in options.items():
        if why:
            say(f"     {e:<13} not here: {why}")
    while True:
        answer = ask("Number or name", "1").lower()
        if answer in usable:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(usable):
            return usable[int(answer) - 1]
        say(f"  Type 1 to {len(usable)}, or a name from the list.")


def choose(name=None, embed=None):
    name = name or pick_provider()
    if name not in PROVIDERS:
        sys.exit(f"Unknown provider '{name}'. Choose one of: {', '.join(PROVIDERS)}.")
    _, chat, judge, key_var = PROVIDERS[name]
    text, added = add_missing(read(TOML, EXAMPLE.name))
    if added:
        say(f"\nproviders.toml was older than the lab: added {', '.join(added)}.")
    old_embed = current_default(text, "embed")

    if key_var:  # first, so the Nexus questions can list the gateway's models
        say("")
        ask_key(key_var)
    if name == "nexus":
        text = ask_nexus(text)
    if name == "ollama":
        have = ollama_models()
        model = get_in_block(text, "ollama", "model")
        if have is None:
            say(f"\nOllama doesn't answer at {OLLAMA}. Install it (Windows: winget install --id Ollama.Ollama -e;\n"
                "macOS: brew install ollama, or the app from ollama.com), start it, and run this again.")
        elif model not in have:
            pull(model)

    embed = pick_embed(name, text, embed)
    if chat:
        text = set_default(text, "chat", chat)
        text = set_default(text, "judge", judge)
    text = set_default(text, "embed", embed)
    TOML.write_text(text, encoding="utf-8")
    say(f"\nproviders.toml now uses chat = {current_default(text, 'chat')}, judge = {current_default(text, 'judge')}, "
        f"embed = {embed}.")
    if old_embed and old_embed != embed and (HERE / ".store").exists():
        say("The embed block changed: an index built with the old one doesn't fit. Run:  python rag.py ingest")
    return name


def check(name):
    py = VENV_PY if VENV_PY.exists() else Path(sys.executable)
    say("\nChecking ...")
    if name == "offline":
        say("Offline: no model to check. Add --dry to every command, for example:\n"
            '  python rag.py ingest --dry\n  python rag.py ask "why did we drop ClearSpring?" --dry')
        return
    subprocess.run([str(py), str(HERE / "rag.py"), "check"])
    # the rulebook dataset embeds with bge-small whatever the default is: fetch it now, not in the middle of the lab
    say("\nThe rulebook (LAB.md) uses bge-small. Fetching it from Hugging Face if it isn't here yet ...")
    out = subprocess.run([str(py), str(HERE / "rag.py"), "check", "--embed", "bge-small"],
                         capture_output=True, text=True).stdout
    line = next((l for l in out.splitlines() if "embed block" in l), out.strip()[-300:])
    say(line)
    if "[ok]" not in line:
        say(HF_HINT)
    say('\nNext:  python rag.py ingest   then   python rag.py ask "why did we drop ClearSpring?"')
    say("Switch models any time with:  python setup.py provider")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] in (["-h"], ["--help"]):
        say(__doc__)
    elif args[:1] == ["provider"] and len(args) <= 3:
        check(choose(*args[1:]))
    elif len(args) <= 2:
        install()
        check(choose(*args))
    else:
        sys.exit(__doc__)
