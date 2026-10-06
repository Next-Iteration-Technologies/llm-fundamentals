import json

from scripted_llm import ScriptedLLM, answers, asks_for
from step2_agent_loop import ARCHIVE_TOOLS, MAX_CALLS_PER_QUESTION, STEP_LIMIT_ANSWER, run_agent_loop


def ask(llm, question):
    history = [{"role": "user", "content": question}]
    return run_agent_loop(llm, history, ARCHIVE_TOOLS, "system"), history


def test_the_tool_result_goes_back_to_the_model():
    llm = ScriptedLLM([asks_for("search_document", doc_id="1234"), answers("It is in Finance.")])
    answer, history = ask(llm, "Where is invoice 1234?")
    assert answer == "It is in Finance."
    tool_message = history[2]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["workspace"] == "Finance"
    assert llm.requests[1]["messages"][-1] == tool_message      # the second call saw the result


def test_the_loop_stops_at_the_step_limit():
    llm = ScriptedLLM([asks_for("search_document", doc_id="1234")] * MAX_CALLS_PER_QUESTION)
    answer, _ = ask(llm, "Where is invoice 1234?")
    assert answer == STEP_LIMIT_ANSWER


def test_an_unknown_tool_becomes_an_error_for_the_model():
    llm = ScriptedLLM([asks_for("delete_everything"), answers("Sorry.")])
    _, history = ask(llm, "Delete it all")
    assert "no tool called delete_everything" in history[2]["content"]


def test_wrong_arguments_become_an_error_for_the_model():
    llm = ScriptedLLM([asks_for("search_document", document="1234"), answers("Sorry.")])
    _, history = ask(llm, "Where is invoice 1234?")
    assert "Wrong arguments" in history[2]["content"]
