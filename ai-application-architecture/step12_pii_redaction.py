"""
Step 12: PII redaction.

Routing personal data to the local model (step 11) helps. Safer still: the
model never sees it. Before every call, names, email addresses and phone
numbers are swapped for placeholders like [PERSON_1]. The same person always
gets the same placeholder, so the model can still reason ("[PERSON_1] and
[PERSON_2] see Finance through Finance-Records"). The mapping stays in our
code for this chat. Names are put back only for users allowed to see them:
developers see everyone, a department user sees only their own name.

With nothing personal left in the text, the gateway can use the company model again.

Try it:
    uv run step12_pii_redaction.py --user mia --demo      # names restored for a developer
    uv run step12_pii_redaction.py --user priya --demo    # Priya's own name only

    You: Hi, I'm Priya Sharma. Where is invoice 1234?
    You: Who can see the Finance workspace?
"""

import re
from collections.abc import Callable
from dataclasses import replace

import llm_client
from step1_bare_model_call import run_safely
from step6_mcp_client import McpConnection
from step8_authentication import Chatbot, Login, User, run_signed_in_chat
from step9_authorization import SECURE_SERVER
from step9_mcp_server_secure import is_developer
from step10_guardrails import check_output
from step11_model_gateway import PERSONAL_DATA
from step11_model_gateway import answer as answer_guarded
from step11_model_gateway import build as build_with_gateway

PLACEHOLDER = re.compile(r"\[(?:" + "|".join(PERSONAL_DATA) + r")_\d+\]")
DEMO_LINES = [
    "Hi, I'm Priya Sharma. Where is invoice 1234?",
    "Who can see the Finance workspace?",
]


class Redactor:
    """Swaps personal data for placeholders, and back. One Redactor per chat."""

    def __init__(self, patterns: dict[str, re.Pattern] = PERSONAL_DATA):
        self._patterns = patterns
        self._placeholder_for: dict[str, str] = {}     # value (lower case) -> placeholder
        self._value_of: dict[str, str] = {}            # placeholder -> value as first seen

    def redact(self, text: str) -> str:
        for kind, pattern in self._patterns.items():
            text = pattern.sub(lambda match, kind=kind: self._placeholder(kind, match.group()), text)
        return text

    def restore(self, text: str, may_see: Callable[[str], bool] = lambda value: True) -> str:
        """Put values back, but only those the reader may see."""
        def value_or_placeholder(match: re.Match) -> str:
            value = self._value_of.get(match.group())
            return value if value is not None and may_see(value) else match.group()

        return PLACEHOLDER.sub(value_or_placeholder, text)

    def _placeholder(self, kind: str, value: str) -> str:
        key = value.lower()
        if key not in self._placeholder_for:
            number = sum(placeholder.startswith(f"[{kind}_") for placeholder in self._value_of) + 1
            placeholder = f"[{kind}_{number}]"
            self._placeholder_for[key] = placeholder
            self._value_of[placeholder] = value
        return self._placeholder_for[key]


def map_text_values(arguments: dict, change: Callable[[str], str]) -> dict:
    return {name: change(value) if isinstance(value, str) else value for name, value in arguments.items()}


class RedactingModel:
    """In front of the model: it sees placeholders, the tools get the real values back."""

    def __init__(self, model, redactor: Redactor):
        self.model = model
        self.redactor = redactor

    def chat(self, messages, system=None, tools=None):
        reply = self.model.chat([self._redacted(message) for message in messages],
                                system=self.redactor.redact(system) if system else system, tools=tools)
        restored_calls = [{**call, "arguments": map_text_values(call["arguments"], self.redactor.restore)}
                          for call in reply.tool_calls]
        return replace(reply, tool_calls=restored_calls)

    def _redacted(self, message: dict) -> dict:
        redacted = {**message, "content": self.redactor.redact(message.get("content") or "")}
        if message.get("tool_calls"):
            redacted["tool_calls"] = [{**call, "arguments": map_text_values(call["arguments"], self.redactor.redact)}
                                      for call in message["tool_calls"]]
        return redacted


def personal_data_visible_to(user: User) -> Callable[[str], bool]:
    """Developers may see everyone's data; anyone else only their own."""
    if is_developer(user):
        return lambda value: True
    own = {user.name.lower(), user.email.lower(), *user.name.lower().split()}
    return lambda value: value.lower() in own


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    bot = build_with_gateway(llm, login, connection)
    return replace(bot, model=RedactingModel(bot.model, Redactor()))


def answer(bot: Chatbot, question: str) -> str:
    reply = answer_guarded(bot, question)
    restored = bot.model.redactor.restore(reply, personal_data_visible_to(bot.login.user))
    return check_output(restored)      # check again: putting values back can bring an outside address back


if __name__ == "__main__":
    run_safely(lambda: run_signed_in_chat("step12_pii_redaction", DEMO_LINES, SECURE_SERVER, build, answer))
