"""
Step 13: logging, traces, evaluation and cost.

"Request answered in 4 s" says nothing when the answer is wrong. A trace
records every step of one question: who asked, each call to the model (which
model, tokens, cost), each tool call with its arguments and result, and the
answer. One JSON line per step, in traces/<date>.jsonl, with personal data
masked before it is written.

The golden questions (data/golden_questions.json) are the regression test:
questions with the tools they must use and the words the answer must (or must
not) contain. Run them after every change to a prompt, a tool or a model.

Try it:
    uv run step13_logging_traces.py --user priya --demo     # chat; then open traces/
    uv run step13_logging_traces.py --golden                # run the golden questions
"""

import argparse
import json
import time
import uuid
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import archive
import llm_client
from request_view import shorten
from step1_bare_model_call import run_safely
from step2_agent_loop import Tool
from step4_conversation_store import ConversationStore
from step6_mcp_client import McpConnection
from step8_authentication import IDENTITY_PROVIDER, Chatbot, Login, run_signed_in_chat
from step9_authorization import SECURE_SERVER
from step12_pii_redaction import Redactor
from step12_pii_redaction import answer as answer_redacted
from step12_pii_redaction import build as build_redacted

TRACE_FOLDER = Path(__file__).parent / "traces"
UNMASKED_FIELDS = {"trace", "time", "kind", "user", "groups"}   # ids, not personal data: needed to explain a trace
DEMO_LINES = [
    "Where is invoice 1234?",
    "I uploaded 20 invoices yesterday. Did they all get archived?",
]


class Tracer:
    """One JSON line per step of a question. Free text is masked with the chat's redactor."""

    def __init__(self, redactor: Redactor, folder: Path = TRACE_FOLDER):
        self._redactor = redactor
        self._file = folder / f"{date.today().isoformat()}.jsonl"
        folder.mkdir(exist_ok=True)
        self.trace_id = None
        self.tools_called: list[str] = []

    @property
    def file_name(self) -> str:
        return self._file.name

    def start(self, login: Login, question: str):
        self.trace_id = uuid.uuid4().hex[:8]
        self.tools_called = []
        self.write("question", user=login.user.user_id, groups=list(login.user.groups), question=question)

    def write(self, kind: str, **fields):
        line = {"trace": self.trace_id, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": kind, **fields}
        masked = {name: value if name in UNMASKED_FIELDS else self._masked(value) for name, value in line.items()}
        with self._file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(masked, ensure_ascii=False) + "\n")

    def _masked(self, value):
        return json.loads(self._redactor.redact(json.dumps(value, ensure_ascii=False)))


def model_prices() -> dict[str, float]:
    """Input price per million tokens, by model name, from providers.toml."""
    return {block["model"]: block.get("input_price_per_million", 0)
            for block in llm_client.load_providers().values() if isinstance(block, dict)}


class TracingModel:
    """In front of the model: every call is written to the trace. Anything else goes to the model."""

    def __init__(self, model, tracer: Tracer):
        self.model = model
        self.tracer = tracer
        self._prices = model_prices()

    def chat(self, messages, system=None, tools=None):
        started = time.perf_counter()
        reply = self.model.chat(messages, system=system, tools=tools)
        input_cost = (reply.input_tokens or 0) * self._prices.get(reply.model, 0) / 1_000_000
        self.tracer.write("model_call", model=reply.model, messages_sent=len(messages),
                          asked_for=[call["name"] for call in reply.tool_calls],
                          input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                          input_cost_usd=round(input_cost, 6), seconds=round(time.perf_counter() - started, 2))
        return reply

    def __getattr__(self, name):
        """Anything else comes from the wrapped model, for example its .redactor."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.model, name)


def traced(tool: Tool, tracer: Tracer) -> Tool:
    def run_and_record(**arguments):
        started = time.perf_counter()
        result = tool.function(**arguments)
        tracer.tools_called.append(tool.name)
        tracer.write("tool_call", tool=tool.name, arguments=arguments, result=shorten(str(result)),
                     seconds=round(time.perf_counter() - started, 2))
        return result

    return replace(tool, function=run_and_record)


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    bot = build_redacted(llm, login, connection)
    tracer = Tracer(bot.model.redactor)
    return replace(bot, model=TracingModel(bot.model, tracer), tools=[traced(tool, tracer) for tool in bot.tools])


def answer(bot: Chatbot, question: str) -> str:
    tracer = bot.model.tracer
    tracer.start(bot.login, question)
    reply = answer_redacted(bot, question)
    tracer.write("answer", answer=reply)
    print(f"  trace {tracer.trace_id} written to traces/{tracer.file_name}")
    return reply


# ---------- the golden questions ----------

def check_case(case: dict, reply: str, tools_called: list[str]) -> list[str]:
    """What went wrong for one golden question; an empty list means it passed."""
    problems = [f"did not use {tool}" for tool in case.get("expect_tools", []) if tool not in tools_called]
    problems += [f"answer lacks '{word}'" for word in case.get("expect_words", []) if word.lower() not in reply.lower()]
    problems += [f"answer contains '{word}'" for word in case.get("avoid_words", []) if word.lower() in reply.lower()]
    return problems


def run_golden_questions():
    llm = llm_client.Session("step13_golden", [])
    cases = archive.read_data("golden_questions.json")
    passed = 0
    with McpConnection(SECURE_SERVER) as connection:
        for number, case in enumerate(cases, start=1):
            login = IDENTITY_PROVIDER.sign_in(case["user"])
            bot = replace(build(llm, login, connection), store=ConversationStore(":memory:"))   # a fresh chat each
            reply = answer(bot, case["question"])
            problems = check_case(case, reply, bot.model.tracer.tools_called)
            passed += not problems
            print(f"{'PASS' if not problems else 'FAIL'}  {number}. {case['user']}: {case['question']}")
            for problem in problems:
                print(f"        {problem}")
    print(f"\n{passed} of {len(cases)} golden questions passed.")


def golden_wanted() -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--golden", action="store_true")
    flags, _ = parser.parse_known_args()
    return flags.golden


if __name__ == "__main__":
    if golden_wanted():
        run_safely(run_golden_questions)
    else:
        run_safely(lambda: run_signed_in_chat("step13_logging_traces", DEMO_LINES, SECURE_SERVER, build, answer))
