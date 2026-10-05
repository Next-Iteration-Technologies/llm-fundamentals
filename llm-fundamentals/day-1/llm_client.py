"""
The plumbing behind every round: which model we talk to, and replay.

The round files stay short because this file holds three things:
  - which provider (Anthropic, Nexus, Ollama) comes from providers.toml
  - --record saves every answer to recordings/<provider>.json
  - --replay answers from those recordings, so nobody needs a key or a network

Flags every round understands:
    --provider NAME   use another block from providers.toml
    --demo            type the round's prepared lines for you
    --record          save the answers while running live
    --replay          answer from the recordings (implies --demo)

Replay looks an answer up by the exact request. If you change a round and the
request is different, there is no recording for it, and replay says so.
"""

import argparse
import atexit
import hashlib
import json
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
RECORDINGS_FOLDER = HERE / "recordings"
MAX_ANSWER_TOKENS = 16000


def load_env_file():
    """Read keys like ANTHROPIC_API_KEY from .env in this folder. A variable already set wins."""
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


@dataclass
class Reply:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    replayed: bool = False


class SetupProblem(Exception):
    """Something on this laptop stops the live call. The message says what to do."""


def load_providers():
    own_file = HERE / "providers.toml"
    config_file = own_file if own_file.exists() else HERE / "providers.example.toml"
    with open(config_file, "rb") as file:
        return tomllib.load(file)


def request_key(kind, **request):
    """A short fingerprint of a request, so replay can find the answer to exactly it."""
    text = json.dumps({"kind": kind, **request}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Session:
    def __init__(self, round_name, demo_lines, default_provider=None):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--provider")
        parser.add_argument("--demo", action="store_true")
        parser.add_argument("--record", action="store_true")
        parser.add_argument("--replay", action="store_true")
        flags, _ = parser.parse_known_args()

        providers = load_providers()
        if flags.replay:
            name = flags.provider or default_provider or providers["replay_from"]
        else:
            name = flags.provider or os.environ.get("LLM_PROVIDER") or default_provider or providers["default"]
        if name not in providers:
            sys.exit(f"No provider called '{name}' in providers.toml.")

        self.round_name = round_name
        self.provider_name = name
        self.provider = providers[name]
        self.replay = flags.replay
        self.record = flags.record and not flags.replay
        self.lines_to_type = list(demo_lines) if (flags.demo or flags.replay) else None
        self.price_per_million = self.provider.get("input_price_per_million", 0)

        self.recordings_file = RECORDINGS_FOLDER / f"{name}.json"
        self.recordings = self._load_recordings()
        self.times_asked = {}      # request key -> how often this session sent it
        self.new_answers = {}      # request key -> answers recorded in this session
        if self.record:
            atexit.register(self._save_recordings)

    # ---------- the user's side ----------

    def read_question(self, prompt="You: "):
        """The next line from the keyboard, or from the prepared lines in demo and replay."""
        if self.lines_to_type is None:
            return input(prompt).strip()
        if not self.lines_to_type:
            return None
        line = self.lines_to_type.pop(0)
        print(f"{prompt}{line}")
        return line

    # ---------- the model's side ----------

    def chat(self, messages, system=None):
        """Send a conversation, get the next assistant message."""
        key = request_key("chat", system=system, messages=messages)
        if self.replay:
            return self._replayed(key)
        reply = self._live_chat(messages, system)
        self._remember(key, reply)
        return reply

    def complete(self, text, temperature, max_new_tokens):
        """Send plain text, no roles, and get the continuation. Needs a base model (Ollama)."""
        key = request_key("complete", text=text, temperature=temperature, max_new_tokens=max_new_tokens)
        if self.replay:
            return self._replayed(key)
        if self.provider["kind"] != "ollama":
            raise SetupProblem(
                f"Plain-text continuation needs a base model. '{self.provider_name}' is a chat API: "
                "it always wraps your text as a user message. Use --provider ollama, or --replay."
            )
        reply = self._ollama_generate(text, temperature, max_new_tokens)
        self._remember(key, reply)
        return reply

    # ---------- live calls, one per kind of API ----------

    def _live_chat(self, messages, system):
        kind = self.provider["kind"]
        if kind == "anthropic":
            return self._anthropic_chat(messages, system)
        if kind == "openai":
            return self._openai_chat(messages, system)
        if kind == "ollama":
            return self._ollama_chat(messages, system)
        raise SetupProblem(f"Unknown kind '{kind}' in providers.toml.")

    def _api_key(self):
        variable = self.provider.get("api_key_env")
        key = os.environ.get(variable, "") if variable else ""
        if variable and not key:
            raise SetupProblem(f"{variable} is not set. Set it, or run with --replay.")
        return key

    def _anthropic_chat(self, messages, system):
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key())
        request = dict(
            model=self.provider["model"],
            max_tokens=MAX_ANSWER_TOKENS,
            messages=messages,
            output_config={"effort": "low"},          # a chat answer, not a hard problem
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                       # a declined request is retried on another model
        )
        if system:
            request["system"] = system
        try:
            response = client.beta.messages.create(**request)
        except anthropic.APIConnectionError:
            raise SetupProblem("Cannot reach the Anthropic API. Check the network, or run with --replay.")
        except anthropic.AuthenticationError:
            raise SetupProblem("The Anthropic key was refused. Check ANTHROPIC_API_KEY, or run with --replay.")

        if response.stop_reason == "refusal":
            text = "(The model declined to answer this one.)"
        else:
            text = "".join(block.text for block in response.content if block.type == "text")
        return Reply(text, response.model, response.usage.input_tokens, response.usage.output_tokens)

    def _openai_chat(self, messages, system):
        import httpx

        all_messages = ([{"role": "system", "content": system}] if system else []) + messages
        try:
            response = httpx.post(
                self.provider["base_url"].rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key()}"},
                json={"model": self.provider["model"], "messages": all_messages},
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise SetupProblem(f"Call to {self.provider_name} failed ({error}). Check providers.toml, or run with --replay.")
        body = response.json()
        usage = body.get("usage", {})
        text = body["choices"][0]["message"]["content"]
        return Reply(text, body.get("model", self.provider["model"]), usage.get("prompt_tokens"), usage.get("completion_tokens"))

    def _ollama_post(self, path, payload):
        import httpx

        try:
            response = httpx.post(self.provider["base_url"].rstrip("/") + path, json=payload, timeout=300)
            response.raise_for_status()
        except httpx.ConnectError:
            raise SetupProblem("Ollama is not running. Start it with 'ollama serve', or run with --replay.")
        except httpx.HTTPStatusError as error:
            raise SetupProblem(f"Ollama answered {error.response.status_code}: {error.response.text.strip()}")
        return response.json()

    def _ollama_chat(self, messages, system):
        all_messages = ([{"role": "system", "content": system}] if system else []) + messages
        body = self._ollama_post("/api/chat", {"model": self.provider["model"], "messages": all_messages, "stream": False})
        return Reply(body["message"]["content"], self.provider["model"], body.get("prompt_eval_count"), body.get("eval_count"))

    def _ollama_generate(self, text, temperature, max_new_tokens):
        body = self._ollama_post("/api/generate", {
            "model": self.provider["model"],
            "prompt": text,
            "raw": True,                     # no chat template: the text goes in exactly as written
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_new_tokens},
        })
        return Reply(body["response"], self.provider["model"], body.get("prompt_eval_count"), body.get("eval_count"))

    # ---------- recordings ----------

    def _load_recordings(self):
        if self.recordings_file.exists():
            return json.loads(self.recordings_file.read_text())
        return {"provider": self.provider_name, "answers": {}}

    def _replayed(self, key):
        answers = self.recordings["answers"].get(key)
        if not answers:
            return Reply("(No recording for this exact request. Run it live to see the answer.)",
                         self.recordings.get("model", "replay"), None, None, replayed=True)
        turn = self.times_asked.get(key, 0)
        self.times_asked[key] = turn + 1
        answer = answers[turn % len(answers)]
        return Reply(answer["text"], answer["model"], answer["input_tokens"], answer["output_tokens"], replayed=True)

    def _remember(self, key, reply):
        if self.record:
            self.new_answers.setdefault(key, []).append({
                "text": reply.text, "model": reply.model,
                "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            })

    def _save_recordings(self):
        if not self.new_answers:
            return
        self.recordings["answers"].update(self.new_answers)
        self.recordings["model"] = self.provider.get("model")
        RECORDINGS_FOLDER.mkdir(exist_ok=True)
        self.recordings_file.write_text(json.dumps(self.recordings, indent=2, ensure_ascii=False))
        print(f"(Recorded {len(self.new_answers)} request(s) to {self.recordings_file.name}.)")
