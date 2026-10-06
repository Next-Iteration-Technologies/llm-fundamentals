"""
Round 2: a chat that remembers.

The LLM itself still has no memory. We keep the whole conversation in a list
and send all of it with every new question. That list IS the memory. Watch the
printout: 1 message, then 3, then 5, and the input tokens rise every call.

Try it:
    uv run r2_chat_with_history.py           # type yourself
    uv run r2_chat_with_history.py --demo    # the prepared lines, live
    uv run r2_chat_with_history.py --replay  # the trainer's recorded answers
    uv run r2_chat_with_history.py --long --demo   # 20 turns, then the cost

    You: My name is Priya and I look after invoices.
    You: What is my name?
    You: What's today's date?

Break the memory: find the line marked THE MEMORY and change `chat_history`
to `chat_history[-1:]` (send only the newest message). Run again.
"""

import sys

import llm_client
from request_view import show_request

DEMO_LINES = [
    "My name is Priya and I look after invoices.",
    "What is my name?",
    "What's today's date?",
]
LONG_CHAT_LINES = [
    "My name is Priya and I look after invoices.",
    "Our team wants a half-day outing next month. Give me three ideas.",
    "We are twelve people. Which idea works best for that size?",
    "Make it cheaper: we have about 40 euros per person.",
    "Two colleagues don't drink alcohol. Does that change anything?",
    "Write a short invitation email for it.",
    "Make the email friendlier and shorter.",
    "Add a line asking people to reply by Friday.",
    "Now write a reminder for the people who haven't replied.",
    "One person asks for leave that day. How should I answer?",
    "Draft that answer in two sentences.",
    "List what I need to book, as a checklist.",
    "Which of those should I book first, and why?",
    "Estimate the total cost for twelve people.",
    "Summarise the plan in five bullet points for my manager.",
    "My manager asks for a backup plan if it rains. Suggest one.",
    "Add the backup plan to the summary.",
    "Write a one-line message for the team chat.",
    "What did we decide about the budget?",
    "What is my name, and what do I look after?",
]
EXIT_WORDS = {"exit", "quit", "bye"}


def run_chat():
    lines = LONG_CHAT_LINES if "--long" in sys.argv else DEMO_LINES
    llm = llm_client.Session("r2_chat_with_history", lines)
    print("Chat with history. Type 'exit' to stop.\n")

    chat_history = []  # grows by two messages every turn: your question, then the answer
    total_input_tokens = 0
    call_number = 0

    while True:
        question = llm.read_question()

        if question is None or question.lower() in EXIT_WORDS:
            print(f"Goodbye! We exchanged {len(chat_history)} messages.")
            break
        if not question:
            continue

        chat_history.append({"role": "user", "content": question})

        messages_to_send = chat_history  # THE MEMORY: everything so far goes with every question

        call_number += 1
        reply = llm.chat(messages_to_send)
        print(f"LLM: {reply.text}\n")
        show_request(call_number, messages_to_send, reply)

        chat_history.append({"role": "assistant", "content": reply.text})
        total_input_tokens += reply.input_tokens or 0

    print_cost(call_number, total_input_tokens, llm.price_per_million)


def print_cost(calls, total_input_tokens, price_per_million):
    if not calls:
        return
    print(f"{calls} calls sent {total_input_tokens} input tokens in total.")
    if price_per_million:
        cost = total_input_tokens * price_per_million / 1_000_000
        print(f"At ${price_per_million:.2f} per million input tokens that is ${cost:.4f}, for one person's chat.")


if __name__ == "__main__":
    try:
        run_chat()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")
