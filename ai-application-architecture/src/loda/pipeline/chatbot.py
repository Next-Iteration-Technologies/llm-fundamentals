from dataclasses import dataclass

from loda.core.auth import Login
from loda.core.model import LLMModel
from loda.core.store import ConversationStore
from loda.core.tools import Tool

IN_MEMORY_STORE = ":memory:"


@dataclass(frozen=True)
class Chatbot:
    """Everything one signed-in chat needs. Behaviour is composed by the pipeline."""
    llm: LLMModel
    login: Login
    tools: list[Tool]
    system: str
    store: ConversationStore
    model: LLMModel | None = None   # wraps llm when a gateway / redactor / tracer is active

    @property
    def session_id(self) -> str:
        return f"{self.login.user.user_id}-chat"

    @property
    def tools_called(self) -> list[str]:
        """Delegated to the model layer when a tracer is present."""
        return getattr(self.model, "tools_called", [])
