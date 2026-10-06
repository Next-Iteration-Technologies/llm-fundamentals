"""
Round 1: a stateless chat.

Every question is sent to the LLM on its own. The model never sees what you
asked before, so it cannot remember anything between turns. The printout after
each answer shows it: one message, every time.

Try it:
    uv run r1_chat_stateless.py              # type yourself
    uv run r1_chat_stateless.py --demo       # the prepared lines, live
    uv run r1_chat_stateless.py --replay     # the trainer's recorded answers

    You: My name is Priya and I look after invoices.
    You: What is my name?
    You: What's today's date?
"""

import sys

import llm_client
from request_view import show_request

DEMO_LINES = [
    "My name is Priya and I look after invoices.",
    "What is my name?",
    "What's today's date?",
]
EXIT_WORDS = {"exit", "quit", "bye"}


def run_chat():
    llm = llm_client.Session("r1_chat_stateless", DEMO_LINES)
    print("Stateless chat. Type 'exit' to stop.\n")
    call_number = 0

    while True:
        question = llm.read_question()

        if question is None or question.lower() in EXIT_WORDS:
            print("Goodbye!")
            break
        if not question:
            continue

        messages = [{"role": "user", "content": question}]  # only the new question, every time

        call_number += 1
        reply = llm.chat(messages)
        print(f"LLM: {reply.text}\n")
        show_request(call_number, messages, reply)


if __name__ == "__main__":
    try:
        run_chat()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")
