"""Verify that concrete stub classes satisfy every Protocol defined in loda.core."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from llm_client import Reply
from loda.core.auth import Identity, Login, NotSignedIn, User
from loda.core.model import LLMModel
from loda.core.store import ConversationStore
from loda.core.tools import Tool, arguments_schema, text_field
from loda.core.tracer import Tracer
from loda.pipeline.chatbot import Chatbot, IN_MEMORY_STORE


# ---------- minimal stubs that implement each Protocol ----------

class _StubModel:
    def chat(self, messages, system=None, tools=None):
        return Reply(text="ok", model="stub", input_tokens=1, output_tokens=1)


class _StubStore:
    def load(self, session_id: str) -> list[dict]:
        return []

    def save(self, session_id: str, history: list[dict]) -> None:
        pass

    def forget(self, session_id: str) -> None:
        pass


class _StubIdentity:
    def sign_in(self, user_id: str) -> Login:
        user = User(user_id, "Test", "test@corp", ("Finance",))
        return Login(user, "tok")

    def verify(self, token: str) -> User:
        return User("priya", "Priya", "priya@corp", ("Finance",))


class _StubTracer:
    @property
    def tools_called(self) -> list[str]:
        return []

    def start(self, login: Login, question: str) -> None:
        pass

    def write(self, kind: str, **fields: object) -> None:
        pass


# ---------- Protocol conformance ----------

def test_stub_model_satisfies_llm_model_protocol():
    assert isinstance(_StubModel(), LLMModel)


def test_stub_store_satisfies_conversation_store_protocol():
    assert isinstance(_StubStore(), ConversationStore)


def test_stub_identity_satisfies_identity_protocol():
    assert isinstance(_StubIdentity(), Identity)


def test_stub_tracer_satisfies_tracer_protocol():
    assert isinstance(_StubTracer(), Tracer)


# ---------- Tool ----------

def test_tool_definition_excludes_function():
    tool = Tool(
        name="search",
        description="finds things",
        parameters=arguments_schema({"doc_id": text_field("id")}),
        function=lambda doc_id: {},
    )
    defn = tool.definition()
    assert defn == {"name": "search", "description": "finds things", "parameters": tool.parameters}
    assert "function" not in defn


def test_arguments_schema_separates_required_and_optional():
    schema = arguments_schema(
        {"doc_id": text_field("id")},
        {"caller_token": text_field("token")},
    )
    assert "doc_id" in schema["required"]
    assert "caller_token" not in schema["required"]
    assert "caller_token" in schema["properties"]


# ---------- Auth primitives ----------

def test_not_signed_in_is_exception():
    assert issubclass(NotSignedIn, Exception)


def test_user_is_immutable():
    user = User("priya", "Priya", "priya@corp", ("Finance",))
    try:
        user.user_id = "arun"   # type: ignore[misc]
        assert False, "should have raised"
    except (AttributeError, TypeError):
        pass


# ---------- Chatbot ----------

def _make_chatbot() -> Chatbot:
    user = User("priya", "Priya", "priya@corp", ("Finance",))
    login = Login(user, "tok")
    return Chatbot(
        llm=_StubModel(),
        login=login,
        tools=[],
        system="You are LoDA.",
        store=_StubStore(),
    )


def test_chatbot_session_id_uses_user_id():
    bot = _make_chatbot()
    assert bot.session_id == "priya-chat"


def test_chatbot_tools_called_is_empty_without_tracer():
    bot = _make_chatbot()
    assert bot.tools_called == []


def test_in_memory_store_constant():
    assert IN_MEMORY_STORE == ":memory:"
