"""
Step 1: a bare model call.

The smallest LoDA chatbot: a system prompt, the user's question, one call to
the model. Nothing else. The model has never seen our archive, so it can only
guess, and it guesses with confidence. Watch the printout: one message, no
tools, nothing it could check. Ask the second question: it has also forgotten
the first one.

Try it:
    uv run step1_bare_model_call.py              # type yourself
    uv run step1_bare_model_call.py --demo       # the prepared lines, live
    uv run step1_bare_model_call.py --replay     # the trainer's recorded answers

    You: Hi, I'm Priya. I can't find my invoice 1234.
    You: Is it in the 2024 folder?

This file also holds the small helpers every later step reuses: reading the
user's questions, numbering the printouts, and stopping cleanly on a setup problem.
"""

import itertools
import sys
from collections.abc import Callable, Iterator

import llm_client
from request_view import show_request

SYSTEM_PROMPT = (
    "You are the LoDA chatbot. You help department users with questions about the LoDA "
    "document archive (CSP and Classic). Answer briefly and in plain language."
)
DEMO_LINES = [
    "Hi, I'm Priya. I can't find my invoice 1234.",
    "Is it in the 2024 folder?",
]
EXIT_WORDS = {"exit", "quit", "bye"}

_call_numbers = itertools.count(1)


# ---------- helpers every step reuses ----------

def read_questions(llm: llm_client.Session) -> Iterator[str]:
    """The user's questions, one at a time, until they type exit."""
    while True:
        question = llm.read_question()
        if question is None or question.lower() in EXIT_WORDS:
            print("Goodbye!")
            return
        if question:
            yield question


def show_call(messages, reply, system=None, tools=None):
    """The printout after every call to the model, numbered across the whole run."""
    show_request(next(_call_numbers), messages, reply, system=system, tools=tools)


def run_safely(chat: Callable[[], None]):
    """Run a chat; a setup problem (no key, no network) ends it with advice instead of a stack trace."""
    try:
        chat()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")


# ---------- this step ----------

def ask_model(llm: llm_client.Session, question: str) -> str:
    """One question, one call. Only the new question is sent: the model sees nothing else."""
    messages = [{"role": "user", "content": question}]
    reply = llm.chat(messages, system=SYSTEM_PROMPT)
    show_call(messages, reply, system=SYSTEM_PROMPT)
    return reply.text


def run_chat():
    llm = llm_client.Session("step1_bare_model_call", DEMO_LINES)
    print(f"LoDA chatbot, a bare model call on {llm.provider_name}. Type 'exit' to stop.\n")
    for question in read_questions(llm):
        print(f"LLM: {ask_model(llm, question)}\n")


if __name__ == "__main__":
    run_safely(run_chat)
