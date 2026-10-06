"""
Round 2b: a chat that keeps only the last few messages.

Sending everything forever gets slow and expensive, and one day it no longer
fits in the context window. So real apps cut. This one sends only the newest
question plus the two exchanges before it. Whatever falls off the front is gone:
the model never saw it.

Try it:
    uv run r2b_chat_keep_last.py --demo      # the prepared lines, live
    uv run r2b_chat_keep_last.py --replay    # the trainer's recorded answers

    You: My name is Priya and I look after invoices.
    You: Give me one tip for a clear email subject line.
    You: What's a good way to start a team meeting?
    You: What is my name?

If it still knows the name, look at the printout: is "Priya" hiding in one of
the answers that was still sent?
"""

import sys

import llm_client
from request_view import show_request

KEEP_LAST_MESSAGES = 5  # the new question + the last two question-and-answer pairs
DEMO_LINES = [
    "My name is Priya and I look after invoices.",
    "Give me one tip for a clear email subject line.",
    "What's a good way to start a team meeting?",
    "What is my name?",
]
EXIT_WORDS = {"exit", "quit", "bye"}


def run_chat():
    llm = llm_client.Session("r2b_chat_keep_last", DEMO_LINES)
    print(f"Chat that keeps the last {KEEP_LAST_MESSAGES} messages. Type 'exit' to stop.\n")

    chat_history = []
    call_number = 0

    while True:
        question = llm.read_question()

        if question is None or question.lower() in EXIT_WORDS:
            print(f"Goodbye! We exchanged {len(chat_history)} messages.")
            break
        if not question:
            continue

        chat_history.append({"role": "user", "content": question})

        messages_to_send = chat_history[-KEEP_LAST_MESSAGES:]  # the cut: older messages are not sent
        dropped = len(chat_history) - len(messages_to_send)

        call_number += 1
        reply = llm.chat(messages_to_send)
        print(f"LLM: {reply.text}\n")
        show_request(call_number, messages_to_send, reply, dropped=dropped)

        chat_history.append({"role": "assistant", "content": reply.text})


if __name__ == "__main__":
    try:
        run_chat()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")
