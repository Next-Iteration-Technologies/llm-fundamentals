"""
A browser UI for the LoDA chatbot steps: chat on the left, the request/tool
printout on the right, live.

This file does not change step1/2/3: it imports their system prompts, tools
and constants, and re-runs the same loop they print to the console, emitting
each step as a JSON event instead. The step files stay exactly as the course
teaches them.
"""

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

import llm_client
import step1_bare_model_call as step1
import step2_agent_loop as step2
import step3_retrieval as step3
from request_view import message_text, shorten

HERE = Path(__file__).parent
STATIC_DIR = HERE / "web_static"

STEPS = {
    "1": {"system": step1.SYSTEM_PROMPT, "tools": None},
    "2": {"system": step2.SYSTEM_PROMPT, "tools": step2.ARCHIVE_TOOLS},
    "3": {"system": step3.SYSTEM_PROMPT, "tools": step3.TOOLS},
}

app = FastAPI()
llm = llm_client.Session("web_app", demo_lines=[])
histories: dict[str, list[dict]] = {"1": [], "2": [], "3": []}
call_numbers = {"1": 0, "2": 0, "3": 0}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return {"provider": llm.provider_name, "model": llm.provider.get("model")}


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


def run_step(step: str, user_text: str):
    """Run one question through a step's loop, yielding one event dict per step of the loop."""
    history = histories[step]
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


@app.post("/api/send")
def send(body: dict):
    step, message = body["step"], body["message"]

    def stream():
        try:
            for event in run_step(step, message):
                yield json.dumps(event) + "\n"
        except llm_client.SetupProblem as problem:
            yield json.dumps({"type": "error", "text": str(problem)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/reset")
def reset(body: dict):
    step = body["step"]
    histories[step] = []
    call_numbers[step] = 0
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
