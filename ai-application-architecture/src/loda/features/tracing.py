"""Tracing feature: per-request trace records with cost and tool tracking."""

import json
import logging
import time
import uuid
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from request_view import shorten

import llm_client
from loda.core.auth import Login
from loda.core.tools import Tool
from loda.features.pii import Redactor

logger = logging.getLogger(__name__)

TRACE_FOLDER = Path(__file__).parents[4] / "traces"
_UNMASKED_FIELDS = {"trace", "time", "kind", "user", "groups"}


def _model_prices() -> dict[str, float]:
    return {
        block["model"]: block.get("input_price_per_million", 0)
        for block in llm_client.load_providers().values()
        if isinstance(block, dict)
    }


class Tracer:
    """Writes one JSON line per step of a question; masks personal data before writing."""

    def __init__(self, redactor: Redactor, folder: Path = TRACE_FOLDER):
        self._redactor = redactor
        self._file = folder / f"{date.today().isoformat()}.jsonl"
        folder.mkdir(exist_ok=True)
        self.trace_id: str | None = None
        self.tools_called: list[str] = []

    @property
    def file_name(self) -> str:
        return self._file.name

    def start(self, login: Login, question: str) -> None:
        self.trace_id = uuid.uuid4().hex[:8]
        self.tools_called = []
        self.write(
            "question",
            user=login.user.user_id,
            groups=list(login.user.groups),
            question=question,
        )

    def write(self, kind: str, **fields: object) -> None:
        line = {
            "trace": self.trace_id,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            **fields,
        }
        masked = {
            name: value if name in _UNMASKED_FIELDS else self._masked(value)
            for name, value in line.items()
        }
        with self._file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(masked, ensure_ascii=False) + "\n")

    def _masked(self, value: object) -> object:
        return json.loads(self._redactor.redact(json.dumps(value, ensure_ascii=False)))


class TracingModel:
    """Wraps a model and records every call in the trace."""

    def __init__(self, model, tracer: Tracer):
        self.model = model
        self.tracer = tracer
        self._prices = _model_prices()

    def chat(self, messages: list[dict], system: str | None = None, tools: list[dict] | None = None):
        started = time.perf_counter()
        reply = self.model.chat(messages, system=system, tools=tools)
        input_cost = (reply.input_tokens or 0) * self._prices.get(reply.model, 0) / 1_000_000
        self.tracer.write(
            "model_call",
            model=reply.model,
            messages_sent=len(messages),
            asked_for=[call["name"] for call in reply.tool_calls],
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            input_cost_usd=round(input_cost, 6),
            seconds=round(time.perf_counter() - started, 2),
        )
        return reply

    @property
    def redactor(self) -> Redactor:
        return self.model.redactor


def traced(tool: Tool, tracer: Tracer) -> Tool:
    def run_and_record(**arguments):
        started = time.perf_counter()
        result = tool.function(**arguments)
        tracer.tools_called.append(tool.name)
        tracer.write(
            "tool_call",
            tool=tool.name,
            arguments=arguments,
            result=shorten(str(result)),
            seconds=round(time.perf_counter() - started, 2),
        )
        return result

    return replace(tool, function=run_and_record)
