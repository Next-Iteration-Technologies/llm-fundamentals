from typing import Protocol, runtime_checkable

from llm_client import Reply


@runtime_checkable
class LLMModel(Protocol):
    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> Reply: ...
