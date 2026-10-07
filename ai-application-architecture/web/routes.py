"""The FastAPI app: serves the page and streams each step's events to it."""

import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

import llm_client

from .config import STATIC_DIR, call_numbers, histories, llm
from .events import run_step

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return {"provider": llm.provider_name, "model": llm.provider.get("model")}


@app.post("/api/send")
def send(body: dict):
    step, message = body["step"], body["message"]

    def stream():
        try:
            for event in run_step(step, histories[step], message):
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
