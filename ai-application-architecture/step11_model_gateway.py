"""
Step 11: a model gateway.

Until now every message went to one model, whatever it contained. The gateway
is one door in front of two models:
  - the company model (the default provider, for example Nexus) for ordinary questions,
  - a local model (Ollama, inside our network) for anything with personal data.
If the company model fails, the gateway falls back to the local one. Personal
data never falls back the other way.

The chatbot does not notice: the gateway has the same .chat() as a model.
Watch the line "gateway -> ..." before every call.

Try it (needs Ollama running with the model in providers.toml [ollama-chat]):
    uv run step11_model_gateway.py --user mia --demo

    You: Where is invoice 1234?                      -> the company model
    You: Who can see the Finance workspace?          -> the local model (names and emails)
"""

import re
from dataclasses import replace

import archive
import llm_client
from step1_bare_model_call import run_safely
from step6_mcp_client import McpConnection
from step8_authentication import Chatbot, Login, run_signed_in_chat
from step9_authorization import SECURE_SERVER
from step10_guardrails import EMAIL_ADDRESS, answer
from step10_guardrails import build as build_guarded

PRIVATE_PROVIDER = "ollama-chat"      # a block in providers.toml: the model inside our network
DEMO_LINES = [
    "Where is invoice 1234?",
    "Who can see the Finance workspace?",
]


def known_person_names() -> list[str]:
    """Full, first and last names of everyone in users.json, longest first."""
    names = {part for user in archive.read_data("users.json")
             for part in (user["name"], *user["name"].split())}
    return sorted(names, key=len, reverse=True)


def name_pattern(names: list[str]) -> re.Pattern:
    """Matches a name, also inside a file path like /HR/Sharma_Priya/, but not inside another word."""
    alternatives = "|".join(re.escape(name) for name in names)
    return re.compile(rf"(?<![A-Za-z])(?:{alternatives})(?![A-Za-z])", re.IGNORECASE)


PERSONAL_DATA = {      # kind -> pattern; emails first, so the name inside an address goes with it
    "EMAIL": EMAIL_ADDRESS,
    "PHONE": re.compile(r"(?:\+|\b0)\d[\d /-]{7,}\d"),
    "PERSON": name_pattern(known_person_names()),
}


def contains_personal_data(text: str) -> bool:
    return any(pattern.search(text) for pattern in PERSONAL_DATA.values())


def conversation_text(messages: list[dict], system: str | None) -> str:
    return "\n".join([system or "", *(message.get("content") or "" for message in messages)])


class ModelGateway:
    """One door, two models. Same .chat() as a model, so the agent loop does not notice."""

    def __init__(self, default: llm_client.Session, private: llm_client.Session):
        self.default = default
        self.private = private

    def chat(self, messages, system=None, tools=None):
        if contains_personal_data(conversation_text(messages, system)):
            return self._send(self.private, "personal data stays inside", messages, system, tools)
        try:
            return self._send(self.default, "no personal data", messages, system, tools)
        except llm_client.SetupProblem as problem:
            print(f"  gateway: {self.default.provider_name} failed ({problem})")
            return self._send(self.private, "fallback", messages, system, tools)

    @staticmethod
    def _send(route: llm_client.Session, reason: str, messages, system, tools):
        print(f"  gateway -> {route.provider_name} ({reason})")
        return route.chat(messages, system=system, tools=tools)


def private_model(llm: llm_client.Session) -> llm_client.Session:
    """The model inside our network, recorded and replayed like the default one."""
    return llm_client.Session(f"{llm.round_name}-private", [], provider_name=PRIVATE_PROVIDER)


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    bot = build_guarded(llm, login, connection)
    return replace(bot, model=ModelGateway(default=llm, private=private_model(llm)))


if __name__ == "__main__":
    run_safely(lambda: run_signed_in_chat("step11_model_gateway", DEMO_LINES, SECURE_SERVER, build, answer))
