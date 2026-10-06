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
Tool results are recorded too, so a replay shows the date and the weather of
the day it was recorded.

Messages use one format for every provider, and this file translates:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [{"id", "name", "arguments"}]}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "the result"}
A tool is described as {"name", "description", "parameters": a JSON schema}.
"""

import argparse
import atexit
import hashlib
import json
import os
import sys
import tomllib
from dataclasses import dataclass, field
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
    tool_calls: list = field(default_factory=list)   # [{"id", "name", "arguments"}]: tools the model asks us to run


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
    def __init__(self, round_name, demo_lines, default_provider=None, replay_provider=None, provider_name=None):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--provider")
        parser.add_argument("--demo", action="store_true")
        parser.add_argument("--record", action="store_true")
        parser.add_argument("--replay", action="store_true")
        flags, _ = parser.parse_known_args()

        providers = load_providers()
        if provider_name:              # a fixed route, for example the private model in step 11
            name = provider_name
        elif flags.replay:
            name = flags.provider or replay_provider or default_provider or providers["replay_from"]
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
        self.new_tool_results = {} # tool call key -> results recorded in this session
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

    def chat(self, messages, system=None, tools=None):
        """Send a conversation (and the tools the model may ask for), get the next assistant message."""
        key = request_key("chat", system=system, messages=messages, **({"tools": tools} if tools else {}))
        if self.replay:
            return self._replayed(key)
        reply = self._live_chat(messages, system, tools)
        self._remember(key, reply)
        return reply

    def run_tool(self, call, function):
        """Run the tool the model asked for, on this laptop. Replay gives back the recorded result."""
        key = request_key("tool", name=call["name"], arguments=call["arguments"])
        if self.replay:
            results = self.recordings.get("tool_results", {}).get(key)
            if not results:
                return "(No recording for this tool call. Run it live to see the result.)"
            turn = self.times_asked.get(key, 0)
            self.times_asked[key] = turn + 1
            return results[turn % len(results)]
        result = function(**call["arguments"])
        if self.record:
            self.new_tool_results.setdefault(key, []).append(result)
        return result

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

    def _live_chat(self, messages, system, tools):
        kind = self.provider["kind"]
        if kind == "anthropic":
            return self._anthropic_chat(messages, system, tools)
        if kind == "openai":
            return self._openai_chat(messages, system, tools)
        if kind == "ollama":
            return self._ollama_chat(messages, system, tools)
        raise SetupProblem(f"Unknown kind '{kind}' in providers.toml.")

    def _api_key(self):
        variable = self.provider.get("api_key_env")
        key = os.environ.get(variable, "") if variable else ""
        if variable and not key:
            raise SetupProblem(f"{variable} is not set. Set it, or run with --replay.")
        return key

    def _anthropic_chat(self, messages, system, tools):
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key())
        request = dict(
            model=self.provider["model"],
            max_tokens=MAX_ANSWER_TOKENS,
            messages=anthropic_messages(messages),
            output_config={"effort": "low"},          # a chat answer, not a hard problem
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                       # a declined request is retried on another model
        )
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools
            ]
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
        tool_calls = [
            {"id": block.id, "name": block.name, "arguments": block.input}
            for block in response.content if block.type == "tool_use"
        ]
        return Reply(text, response.model, response.usage.input_tokens, response.usage.output_tokens,
                     tool_calls=tool_calls)

    def _openai_chat(self, messages, system, tools):
        import httpx

        all_messages = ([{"role": "system", "content": system}] if system else []) + openai_messages(messages)
        request = {"model": self.provider["model"], "messages": all_messages}
        if tools:
            request["tools"] = openai_tools(tools)
        key = self._api_key()
        try:
            response = httpx.post(
                self.provider["base_url"].rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"} if key else {},
                json=request,
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise SetupProblem(f"Call to {self.provider_name} failed ({error}). Check providers.toml, or run with --replay.")
        body = response.json()
        usage = body.get("usage", {})
        message = body["choices"][0]["message"]
        tool_calls = [
            {"id": call["id"], "name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"] or "{}")}
            for call in message.get("tool_calls") or []
        ]
        return Reply(message.get("content") or "", body.get("model", self.provider["model"]),
                     usage.get("prompt_tokens"), usage.get("completion_tokens"), tool_calls=tool_calls)

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

    def _ollama_chat(self, messages, system, tools):
        all_messages = ([{"role": "system", "content": system}] if system else []) + ollama_messages(messages)
        request = {"model": self.provider["model"], "messages": all_messages, "stream": False}
        if tools:
            request["tools"] = openai_tools(tools)    # Ollama takes the OpenAI tool format
        if "think" in self.provider:
            request["think"] = self.provider["think"]
        body = self._ollama_post("/api/chat", request)
        message = body["message"]
        tool_calls = [
            {"id": f"call_{number}", "name": call["function"]["name"], "arguments": call["function"]["arguments"]}
            for number, call in enumerate(message.get("tool_calls") or [], start=1)
        ]
        return Reply(message.get("content", ""), self.provider["model"], body.get("prompt_eval_count"),
                     body.get("eval_count"), tool_calls=tool_calls)

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
        return Reply(answer["text"], answer["model"], answer["input_tokens"], answer["output_tokens"],
                     replayed=True, tool_calls=answer.get("tool_calls", []))

    def _remember(self, key, reply):
        if self.record:
            self.new_answers.setdefault(key, []).append({
                "text": reply.text, "model": reply.model,
                "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                **({"tool_calls": reply.tool_calls} if reply.tool_calls else {}),
            })

    def _save_recordings(self):
        if not self.new_answers:
            return
        self.recordings["answers"].update(self.new_answers)
        if self.new_tool_results:
            self.recordings.setdefault("tool_results", {}).update(self.new_tool_results)
        self.recordings["model"] = self.provider.get("model")
        RECORDINGS_FOLDER.mkdir(exist_ok=True)
        self.recordings_file.write_text(json.dumps(self.recordings, indent=2, ensure_ascii=False))
        print(f"(Recorded {len(self.new_answers)} request(s) to {self.recordings_file.name}.)")


# ---------- one message format, translated for each API ----------

def anthropic_messages(messages):
    """Tool calls become tool_use blocks; tool results go back inside a user message."""
    translated = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            blocks = [{"type": "text", "text": message["content"]}] if message["content"] else []
            blocks += [{"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]}
                       for call in message["tool_calls"]]
            translated.append({"role": "assistant", "content": blocks})
        elif message["role"] == "tool":
            result = {"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}
            if translated and translated[-1]["role"] == "user" and isinstance(translated[-1]["content"], list):
                translated[-1]["content"].append(result)     # several results from one turn share a message
            else:
                translated.append({"role": "user", "content": [result]})
        else:
            translated.append({"role": message["role"], "content": message["content"]})
    return translated


def openai_messages(messages):
    translated = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            translated.append({"role": "assistant", "content": message["content"] or None, "tool_calls": [
                {"id": call["id"], "type": "function",
                 "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}
                for call in message["tool_calls"]
            ]})
        elif message["role"] == "tool":
            translated.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]})
        else:
            translated.append({"role": message["role"], "content": message["content"]})
    return translated


def ollama_messages(messages):
    translated = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            translated.append({"role": "assistant", "content": message["content"], "tool_calls": [
                {"function": {"name": call["name"], "arguments": call["arguments"]}} for call in message["tool_calls"]
            ]})
        elif message["role"] == "tool":
            translated.append({"role": "tool", "tool_name": message["name"], "content": message["content"]})
        else:
            translated.append({"role": message["role"], "content": message["content"]})
    return translated


def openai_tools(tools):
    return [{"type": "function", "function": t} for t in tools]
