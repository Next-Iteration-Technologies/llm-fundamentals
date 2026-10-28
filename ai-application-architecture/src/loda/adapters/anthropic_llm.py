"""Wraps llm_client.Session so it satisfies the LLMModel protocol."""

import llm_client
from loda.core.model import LLMModel  # noqa: F401 — imported for re-export and type checking


class AnthropicLLM:
    """Thin adapter over llm_client.Session. Delegates chat() and exposes provider_name."""

    def __init__(self, session: llm_client.Session):
        self._session = session

    def chat(self, messages: list[dict], system: str | None = None, tools: list[dict] | None = None):
        return self._session.chat(messages, system=system, tools=tools)

    @property
    def provider_name(self) -> str:
        return self._session.provider_name

    def run_tool(self, call: dict, executor):
        return self._session.run_tool(call, executor)
