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
from dataclasses import replace

import archive
import llm_client
from step1_bare_model_call import run_safely
from step8_authentication import IDENTITY_PROVIDER, run_signed_in_chat

from loda.adapters.mcp_tools import DEFAULT_SERVER as SECURE_SERVER, McpConnection, tools_from_server
from loda.adapters.sqlite_store import ConversationStore
from loda.core.auth import Login
from loda.features.gateway import ModelGateway, private_model
from loda.features.guardrails import (
    BLOCKED_INPUT_ANSWER, SCOPE_RULES, check_output, looks_like_injection, wrap_as_outside_data,
)
from loda.features.pii import Redactor, RedactingModel, personal_data_visible_to
from loda.features.retrieval import SYSTEM_PROMPT as STEP3_PROMPT
from loda.features.tracing import Tracer, TracingModel, traced
from loda.pipeline.agent import answer_with_memory
from loda.pipeline.builder import acting_as, offered_tools, with_token
from loda.pipeline.chatbot import Chatbot

DEMO_LINES = [
    "Where is invoice 1234?",
    "I uploaded 20 invoices yesterday. Did they all get archived?",
]


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    system = STEP3_PROMPT + f" The signed-in user belongs to: {', '.join(login.user.groups)}." + SCOPE_RULES
    tools = [wrap_as_outside_data(t) for t in
             with_token(offered_tools(acting_as(login.user, tools_from_server(connection)), login.user), login)]
    redactor = Redactor()
    gateway = ModelGateway(default=llm, private=private_model(llm))
    tracer = Tracer(redactor)
    model = TracingModel(RedactingModel(gateway, redactor), tracer)
    return Chatbot(llm, login, [traced(t, tracer) for t in tools], system, ConversationStore(), model=model)


def answer(bot: Chatbot, question: str) -> str:
    if looks_like_injection(question):
        return BLOCKED_INPUT_ANSWER
    tracer = bot.model.tracer
    tracer.start(bot.login, question)
    IDENTITY_PROVIDER.verify(bot.login.token)
    reply = answer_with_memory(bot.llm, bot.store, bot.session_id, question,
                               tools=bot.tools, system=bot.system, model=bot.model)
    is_dev = lambda u: archive.ADMIN_GROUP in u.groups
    restored = bot.model.redactor.restore(reply, personal_data_visible_to(bot.login.user, is_dev))
    result = check_output(restored)
    tracer.write("answer", answer=result)
    print(f"  trace {tracer.trace_id} written to traces/{tracer.file_name}")
    return result


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
            bot = replace(build(llm, login, connection), store=ConversationStore(":memory:"))
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
