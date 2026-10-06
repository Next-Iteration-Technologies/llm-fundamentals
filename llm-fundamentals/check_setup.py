"""
Setup check: live, replay or watch?

Run it once at the start of the day. It checks this laptop step by step and
ends with one word: LIVE (real calls work), REPLAY (use --replay) or WATCH
(follow the trainer's screen).

Try it:
    uv run check_setup.py                    # the default provider
    uv run check_setup.py --provider ollama  # the round 0 model
"""

import os
import sys

import llm_client


def check(label, passed, advice=""):
    print(f"  [{'ok' if passed else '--'}] {label}{'' if passed else '   -> ' + advice}")
    return passed


def provider_reachable(provider):
    import httpx

    kind = provider["kind"]
    try:
        if kind == "anthropic":
            import anthropic
            anthropic.Anthropic(api_key=os.environ[provider["api_key_env"]]).models.retrieve(provider["model"])
            return True, ""
        if kind == "openai":
            headers = {"Authorization": f"Bearer {os.environ[provider['api_key_env']]}"}
            httpx.get(provider["base_url"].rstrip("/") + "/models", headers=headers, timeout=15).raise_for_status()
            return True, ""
        if kind == "ollama":
            try:
                tags = httpx.get(provider["base_url"].rstrip("/") + "/api/tags", timeout=5).json()
            except httpx.ConnectError:
                return False, "Ollama is not running: start it with 'ollama serve'"
            names = [model["name"] for model in tags.get("models", [])]
            if provider["model"] not in names:
                return False, f"run: ollama pull {provider['model']}"
            return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return False, f"unknown kind '{kind}'"


def run_check():
    providers = llm_client.load_providers()
    name = sys.argv[sys.argv.index("--provider") + 1] if "--provider" in sys.argv else (
        os.environ.get("LLM_PROVIDER") or providers["default"])
    provider = providers[name]
    print(f"Setup check for provider '{name}' ({provider['model']})\n")

    python_ok = check(f"Python {sys.version.split()[0]}", sys.version_info >= (3, 11), "use: uv run check_setup.py")
    replay_file = llm_client.RECORDINGS_FOLDER / f"{providers['replay_from']}.json"
    replay_ok = check("recordings for --replay", replay_file.exists(), "ask the trainer for the recordings folder")

    key_variable = provider.get("api_key_env")
    key_ok = check(f"{key_variable} is set" if key_variable else "no key needed",
                   not key_variable or bool(os.environ.get(key_variable)), f"set {key_variable}")
    live_ok = False
    if key_ok:
        reachable, advice = provider_reachable(provider)
        live_ok = check(f"{name} answers", reachable, advice)

    print()
    if python_ok and live_ok:
        print("LIVE: run the rounds as they are.")
    elif python_ok and replay_ok:
        print("REPLAY: add --replay to every round.")
    else:
        print("WATCH: follow the trainer's screen; we fix this at the break.")


if __name__ == "__main__":
    run_check()
