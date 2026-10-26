"""Browser UI for the LoDA chatbot: chat on the left, request/tool log on the right."""

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

import llm_client
from loda.adapters.sqlite_store import ConversationStore
from loda.features.retrieval import SYSTEM_PROMPT as RETRIEVAL_PROMPT
from loda.features.retrieval import TOOLS
from loda.pipeline.agent import MAX_CALLS_PER_QUESTION, STEP_LIMIT_ANSWER
from request_view import message_text, shorten

HERE = Path(__file__).parent
STATIC_DIR = HERE / "web_static"
STORE_SESSION_ID = "priya"

_BARE_PROMPT = (
    "You are the LoDA chatbot. You help department users with questions about the LoDA "
    "document archive (CSP and Classic). Answer briefly and in plain language."
)
_AGENT_PROMPT = _BARE_PROMPT + (
    " Use the tools to check the archive and answer only from their results."
    " If the tools cannot answer, hand the question to the developers with a short summary."
)
_ARCHIVE_TOOLS = [t for t in TOOLS if t.name != "search_known_issues"]

STEPS = {
    "1": {"system": _BARE_PROMPT,     "tools": None,          "store": False},
    "2": {"system": _AGENT_PROMPT,    "tools": _ARCHIVE_TOOLS, "store": False},
    "3": {"system": RETRIEVAL_PROMPT, "tools": TOOLS,          "store": False},
    "4": {"system": RETRIEVAL_PROMPT, "tools": TOOLS,          "store": True},
}

app = FastAPI()
llm = llm_client.Session("web_app", demo_lines=[])
histories: dict[str, list[dict]] = {"1": [], "2": [], "3": []}
call_numbers = {"1": 0, "2": 0, "3": 0, "4": 0}


def _call_safely(function, arguments: dict) -> str:
    try:
        result = function(**arguments)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except TypeError as error:
        return json.dumps({"error": f"Wrong arguments: {error}"})


def open_store() -> ConversationStore:
    """Fresh connection per call: sqlite3 connections cannot cross threads."""
    return ConversationStore()


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
        "messages": [{"role": m["role"], "text": message_text(m)} for m in messages],
        "input_tokens": reply.input_tokens,
    }


def run_step(step: str, user_text: str):
    """Run one question through the configured pipeline, yielding one event per step."""
    config = STEPS[step]
    history = open_store().load(STORE_SESSION_ID) if config["store"] else histories[step]
    history.append({"role": "user", "content": user_text})

    def finish(answer_text: str):
        yield {"type": "answer", "text": answer_text}
        if config["store"]:
            open_store().save(STORE_SESSION_ID, history)
            yield {"type": "stored", "session_id": STORE_SESSION_ID, "count": len(history)}

    if config["tools"] is None:
        reply = llm.chat(history, system=config["system"])
        yield request_event(step, history, reply, config["system"], None)
        history.append({"role": "assistant", "content": reply.text})
        yield from finish(reply.text)
        return

    tools = config["tools"]
    definitions = [tool.definition() for tool in tools]
    tools_by_name = {tool.name: tool for tool in tools}

    for _ in range(MAX_CALLS_PER_QUESTION):
        reply = llm.chat(history, system=config["system"], tools=definitions)
        yield request_event(step, history, reply, config["system"], tools)
        history.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
        if not reply.tool_calls:
            yield from finish(reply.text)
            return
        for call in reply.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = json.dumps({"error": f"There is no tool called {call['name']}."})
            else:
                result = llm.run_tool(call, lambda **arguments: _call_safely(tool.function, arguments))
            yield {"type": "tool_call", "name": call["name"], "arguments": call["arguments"], "result": shorten(result)}
            history.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})

    history.append({"role": "assistant", "content": STEP_LIMIT_ANSWER})
    yield from finish(STEP_LIMIT_ANSWER)


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
    if STEPS[step]["store"]:
        open_store().forget(STORE_SESSION_ID)
    else:
        histories[step] = []
    call_numbers[step] = 0
    return {"ok": True}


@app.get("/api/dump")
def dump(step: str):
    if not STEPS[step]["store"]:
        return {"session_id": None, "count": 0, "messages": []}
    messages = open_store().load(STORE_SESSION_ID)
    return {
        "session_id": STORE_SESSION_ID,
        "count": len(messages),
        "messages": [{"role": m["role"], "text": message_text(m)} for m in messages],
        "raw": messages,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
