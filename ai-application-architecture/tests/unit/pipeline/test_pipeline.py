"""Unit tests for loda.pipeline — chatbot data class and agent loop."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import pytest

from llm_client import Reply
from loda.core.auth import Login, User
from loda.core.tools import Tool, arguments_schema, text_field
from loda.pipeline.agent import MAX_CALLS_PER_QUESTION, STEP_LIMIT_ANSWER, answer_with_memory, run_agent_loop
from loda.pipeline.chatbot import Chatbot, IN_MEMORY_STORE


# ---------- fixtures ----------

def _make_user(name="Priya", groups=("Finance",)):
    return User("priya", name, "priya@corp.example", tuple(groups))


def _make_login(user=None):
    return Login(user or _make_user(), token="tok-test")


def _stub_store():
    """In-memory store stub using a plain list."""
    class StubStore:
        def __init__(self):
            self._data = []

        def load(self, _session_id):
            return list(self._data)

        def save(self, _session_id, history):
            self._data[:] = history

        def forget(self, _session_id):
            self._data.clear()

    return StubStore()


def _stub_tool(name="ping", result="pong"):
    return Tool(
        name=name,
        description="test tool",
        parameters=arguments_schema({}),
        function=lambda: result,
    )


class _AnsweringModel:
    """Model that answers on the first call with no tool requests."""
    def __init__(self, text="ok"):
        self._text = text

    def chat(self, messages, system=None, tools=None):
        return Reply(text=self._text, model="stub", input_tokens=1, output_tokens=1)

    def run_tool(self, call, executor):
        return executor()


class _ToolCallingModel:
    """Model that calls one tool then answers."""
    def __init__(self, tool_name, answer="done"):
        self._tool_name = tool_name
        self._answer = answer
        self._turn = 0

    def chat(self, messages, system=None, tools=None):
        self._turn += 1
        if self._turn == 1:
            return Reply(
                text="",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                tool_calls=[{"id": "c1", "name": self._tool_name, "arguments": {}}],
            )
        return Reply(text=self._answer, model="stub", input_tokens=1, output_tokens=1)

    def run_tool(self, call, executor):
        return executor()


# ---------- Chatbot ----------

def test_chatbot_session_id_uses_user_id():
    login = _make_login()
    bot = Chatbot(
        llm=_AnsweringModel(), login=login, tools=[], system="sys",
        store=_stub_store(),
    )
    assert bot.session_id == "priya-chat"


def test_chatbot_tools_called_empty_without_tracer():
    login = _make_login()
    bot = Chatbot(
        llm=_AnsweringModel(), login=login, tools=[], system="sys",
        store=_stub_store(),
    )
    assert bot.tools_called == []


def test_chatbot_tools_called_delegates_to_model():
    class FakeModel:
        tools_called = ["search_document"]
        def chat(self, *a, **k): ...

    login = _make_login()
    bot = Chatbot(
        llm=_AnsweringModel(), login=login, tools=[], system="sys",
        store=_stub_store(), model=FakeModel(),
    )
    assert bot.tools_called == ["search_document"]


def test_chatbot_answer_raises_without_pipeline_fn():
    login = _make_login()
    bot = Chatbot(
        llm=_AnsweringModel(), login=login, tools=[], system="sys",
        store=_stub_store(),
    )
    with pytest.raises(RuntimeError, match="pipeline"):
        bot.answer("hello")


def test_chatbot_answer_delegates_to_pipeline_fn():
    calls = []

    def my_pipeline(bot, question):
        calls.append(question)
        return "answered"

    login = _make_login()
    bot = Chatbot(
        llm=_AnsweringModel(), login=login, tools=[], system="sys",
        store=_stub_store(), pipeline_fn=my_pipeline,
    )
    result = bot.answer("where is invoice 1234?")
    assert result == "answered"
    assert calls == ["where is invoice 1234?"]


# ---------- agent loop ----------

def test_run_agent_loop_returns_text_immediately():
    llm = _AnsweringModel("the answer")
    history = [{"role": "user", "content": "hello"}]
    result = run_agent_loop(llm, history, tools=[], system="sys")
    assert result == "the answer"
    assert history[-1]["role"] == "assistant"


def test_run_agent_loop_executes_tool_then_answers():
    tool = _stub_tool("ping", "pong")
    llm = _ToolCallingModel("ping", "done")
    history = [{"role": "user", "content": "ping me"}]
    result = run_agent_loop(llm, history, tools=[tool], system="sys")
    assert result == "done"
    roles = [m["role"] for m in history]
    assert "tool" in roles


def test_run_agent_loop_returns_step_limit_answer_when_exhausted():
    class InfiniteToolModel:
        def chat(self, messages, system=None, tools=None):
            return Reply(
                text="",
                model="stub",
                input_tokens=1,
                output_tokens=1,
                tool_calls=[{"id": "c1", "name": "ping", "arguments": {}}],
            )

        def run_tool(self, call, executor):
            return executor()

    tool = _stub_tool("ping")
    llm = InfiniteToolModel()
    history = [{"role": "user", "content": "loop"}]
    result = run_agent_loop(llm, history, tools=[tool], system="sys")
    assert result == STEP_LIMIT_ANSWER


def test_run_agent_loop_uses_model_for_chat_not_llm():
    """When model is provided, chat goes through model, tool execution through llm."""
    chat_calls = []

    class TrackingModel:
        def chat(self, messages, system=None, tools=None):
            chat_calls.append(True)
            return Reply(text="via model", model="stub", input_tokens=1, output_tokens=1)

    llm = _AnsweringModel("via llm")
    history = [{"role": "user", "content": "hi"}]
    result = run_agent_loop(llm, history, tools=[], system="sys", model=TrackingModel())
    assert result == "via model"
    assert chat_calls


# ---------- answer_with_memory ----------

def test_answer_with_memory_persists_to_store():
    store = _stub_store()
    llm = _AnsweringModel("stored answer")
    answer_with_memory(llm, store, "s1", "question?", tools=[], system="sys")
    saved = store.load("s1")
    assert any(m.get("role") == "user" and m.get("content") == "question?" for m in saved)
    assert any(m.get("role") == "assistant" for m in saved)


def test_answer_with_memory_loads_prior_history():
    store = _stub_store()
    store.save("s1", [{"role": "user", "content": "first"}, {"role": "assistant", "content": "one"}])
    llm = _AnsweringModel("two")
    answer_with_memory(llm, store, "s1", "second?", tools=[], system="sys")
    saved = store.load("s1")
    assert len(saved) == 4  # first, one, second?, two
