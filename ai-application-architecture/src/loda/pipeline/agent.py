"""Agent loop: the model + tools execution cycle with persistent memory."""

import json
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

MAX_CALLS_PER_QUESTION = 6
STEP_LIMIT_ANSWER = (
    "I could not finish this within my step limit. Please contact the LoDA developers."
)


def _as_text(result: object) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def _call_safely(function: Callable[..., object], arguments: dict) -> str:
    try:
        return _as_text(function(**arguments))
    except TypeError as error:
        return _as_text({"error": f"Wrong arguments: {error}"})


def _run_tool_call(llm, call: dict, tools_by_name: dict) -> dict:
    tool = tools_by_name.get(call["name"])
    if tool is None:
        result = _as_text({"error": f"There is no tool called {call['name']}."})
    else:
        result = llm.run_tool(call, lambda **arguments: _call_safely(tool.function, arguments))
    logger.debug("tool %s -> %.120s", call["name"], result)
    return {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result}


def run_agent_loop(llm, history: list[dict], tools: list, system: str, model=None) -> str:
    """Call the model; while it asks for tools, run them and call again.

    history grows in place: the question, every tool call and result, then the answer.
    model is anything with .chat(); a gateway / redactor / tracer goes there.
    """
    model = model or llm
    definitions = [tool.definition() for tool in tools]
    tools_by_name = {tool.name: tool for tool in tools}

    for _ in range(MAX_CALLS_PER_QUESTION):
        reply = model.chat(history, system=system, tools=definitions)
        history.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
        if not reply.tool_calls:
            return reply.text
        for call in reply.tool_calls:
            history.append(_run_tool_call(llm, call, tools_by_name))

    history.append({"role": "assistant", "content": STEP_LIMIT_ANSWER})
    return STEP_LIMIT_ANSWER


def answer_with_memory(llm, store, session_id: str, question: str, *, tools: list, system: str, model=None) -> str:
    """Load the session, add the question, run the loop, store everything new."""
    history = store.load(session_id)
    history.append({"role": "user", "content": question})
    answer = run_agent_loop(llm, history, tools, system, model)
    store.save(session_id, history)
    return answer
