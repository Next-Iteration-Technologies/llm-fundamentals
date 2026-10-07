"""
The agent loop, re-run for the browser: the same loop step2/step3 print to
the console, except each step of it is yielded as a JSON-able event instead.
"""

import json

import step2_agent_loop as step2
from request_view import message_text, shorten

from .config import STEPS, call_numbers, llm


def request_event(step: str, messages: list[dict], reply, system: str, tools: list | None) -> dict:
    call_numbers[step] += 1
    return {
        "type": "request",
        "call_number": call_numbers[step],
        "model": reply.model,
        "replayed": reply.replayed,
        "system": shorten(system),
        "tools_offered": [tool.name for tool in tools] if tools else [],
        "messages": [{"role": message["role"], "text": message_text(message)} for message in messages],
        "input_tokens": reply.input_tokens,
    }


def run_step(step: str, history: list[dict], user_text: str):
    """Run one question through a step's loop, yielding one event dict per step of the loop."""
    config = STEPS[step]
    history.append({"role": "user", "content": user_text})

    if config["tools"] is None:
        reply = llm.chat(history, system=config["system"])
        yield request_event(step, history, reply, config["system"], None)
        history.append({"role": "assistant", "content": reply.text})
        yield {"type": "answer", "text": reply.text}
        return

    tools = config["tools"]
    definitions = [tool.definition() for tool in tools]
    tools_by_name = {tool.name: tool for tool in tools}

    for _ in range(step2.MAX_CALLS_PER_QUESTION):
        reply = llm.chat(history, system=config["system"], tools=definitions)
        yield request_event(step, history, reply, config["system"], tools)
        history.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
        if not reply.tool_calls:
            yield {"type": "answer", "text": reply.text}
            return
        for call in reply.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = json.dumps({"error": f"There is no tool called {call['name']}."})
            else:
                result = llm.run_tool(call, lambda **arguments: step2.call_safely(tool.function, arguments))
            yield {"type": "tool_call", "name": call["name"], "arguments": call["arguments"], "result": shorten(result)}
            history.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})

    history.append({"role": "assistant", "content": step2.STEP_LIMIT_ANSWER})
    yield {"type": "answer", "text": step2.STEP_LIMIT_ANSWER}
