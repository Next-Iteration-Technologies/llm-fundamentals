"""PII feature: redaction and selective restoration of personal data."""

import re
from collections.abc import Callable
from dataclasses import replace

import archive
from loda.core.auth import User
from loda.features.guardrails import EMAIL_ADDRESS

_PHONE = re.compile(r"(?:\+|\b0)\d[\d /-]{7,}\d")


def _known_person_names() -> list[str]:
    names = {
        part
        for user in archive.read_data("users.json")
        for part in (user["name"], *user["name"].split())
    }
    return sorted(names, key=len, reverse=True)


def _name_pattern(names: list[str]) -> re.Pattern:
    alternatives = "|".join(re.escape(name) for name in names)
    return re.compile(rf"(?<![A-Za-z])(?:{alternatives})(?![A-Za-z])", re.IGNORECASE)


PERSONAL_DATA: dict[str, re.Pattern] = {
    "EMAIL": EMAIL_ADDRESS,
    "PHONE": _PHONE,
    "PERSON": _name_pattern(_known_person_names()),
}

_PLACEHOLDER = re.compile(
    r"\[(?:" + "|".join(PERSONAL_DATA) + r")_\d+\]"
)


class Redactor:
    """Swaps personal data for stable placeholders, and back. One instance per chat."""

    def __init__(self, patterns: dict[str, re.Pattern] = PERSONAL_DATA):
        self._patterns = patterns
        self._placeholder_for: dict[str, str] = {}
        self._value_of: dict[str, str] = {}

    def redact(self, text: str) -> str:
        for kind, pattern in self._patterns.items():
            def _sub(match: re.Match, kind: str = kind) -> str:
                return self._placeholder(kind, match.group())
            text = pattern.sub(_sub, text)
        return text

    def restore(self, text: str, may_see: Callable[[str], bool] = lambda _: True) -> str:
        def value_or_placeholder(match: re.Match) -> str:
            value = self._value_of.get(match.group())
            return value if value is not None and may_see(value) else match.group()

        return _PLACEHOLDER.sub(value_or_placeholder, text)

    def _placeholder(self, kind: str, value: str) -> str:
        key = value.lower()
        if key not in self._placeholder_for:
            count = sum(p.startswith(f"[{kind}_") for p in self._value_of) + 1
            placeholder = f"[{kind}_{count}]"
            self._placeholder_for[key] = placeholder
            self._value_of[placeholder] = value
        return self._placeholder_for[key]


def map_text_values(arguments: dict, change: Callable[[str], str]) -> dict:
    return {
        name: change(value) if isinstance(value, str) else value
        for name, value in arguments.items()
    }


class RedactingModel:
    """Wraps a model: it sees placeholders; tools receive real values."""

    def __init__(self, model, redactor: Redactor):
        self.model = model
        self.redactor = redactor

    def chat(self, messages: list[dict], system: str | None = None, tools: list[dict] | None = None):
        redacted_messages = [self._redacted(m) for m in messages]
        redacted_system = self.redactor.redact(system) if system else system
        reply = self.model.chat(redacted_messages, system=redacted_system, tools=tools)
        restored_calls = [
            {**call, "arguments": map_text_values(call["arguments"], self.redactor.restore)}
            for call in reply.tool_calls
        ]
        return replace(reply, tool_calls=restored_calls)

    def _redacted(self, message: dict) -> dict:
        redacted = {**message, "content": self.redactor.redact(message.get("content") or "")}
        if message.get("tool_calls"):
            redacted["tool_calls"] = [
                {**call, "arguments": map_text_values(call["arguments"], self.redactor.redact)}
                for call in message["tool_calls"]
            ]
        return redacted


def personal_data_visible_to(
    user: User, is_developer: Callable[[User], bool]
) -> Callable[[str], bool]:
    """Developers may see everyone's data; anyone else only their own."""
    if is_developer(user):
        return lambda _: True
    own = {user.name.lower(), user.email.lower(), *user.name.lower().split()}
    return lambda value: value.lower() in own
