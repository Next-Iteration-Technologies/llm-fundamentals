"""
Round 1b: the same chat, another model.

A stateless chat like round 1, but you choose the model. Ask the same question
to two models and compare the answers. The printout after each answer shows
which model answered.

Try it:
    uv run r1b_choose_model.py                              # the provider's own model
    uv run r1b_choose_model.py --model claude-haiku-5-5     # the same chat, another model
    uv run r1b_choose_model.py --provider ollama-chat --model qwen3:8b
    uv run r1b_choose_model.py --demo                       # the prepared lines, live

    You: Which model are you?
    You: /model claude-sonnet-5-5
    You: Which model are you?
"""

import sys

import llm_client
from request_view import show_request

DEMO_LINES = [
    "Which model are you?",
    "A bat and a ball cost 1.10 euros. The bat costs 1 euro more than the ball. What does the ball cost?",
]
EXIT_WORDS = {"exit", "quit", "bye"}
SWITCH_COMMAND = "/model"


def switch_model(llm, question):
    """'/model NAME' changes the model for the next questions."""
    new_model = question.removeprefix(SWITCH_COMMAND).strip()
    if llm.replay:
        print("Switching models needs a live run. Replay plays back what was recorded.\n")
    elif not new_model:
        print(f"Still talking to {llm.provider['model']}. Type '{SWITCH_COMMAND} NAME' to switch.\n")
    else:
        llm.provider["model"] = new_model
        print(f"Now talking to {llm.provider_name} / {new_model}.\n")


def run_chat():
    llm = llm_client.Session("r1b_choose_model", DEMO_LINES)
    llm.provider = dict(llm.provider)    # our own copy, so /model changes only this chat
    print(f"Talking to {llm.provider_name} / {llm.provider['model']}.")
    print(f"Type '{SWITCH_COMMAND} NAME' to switch models, 'exit' to stop.\n")
    call_number = 0

    while True:
        question = llm.read_question()

        if question is None or question.lower() in EXIT_WORDS:
            print("Goodbye!")
            break
        if not question:
            continue
        if question.startswith(SWITCH_COMMAND):
            switch_model(llm, question)
            continue

        messages = [{"role": "user", "content": question}]  # only the new question, every time

        try:
            reply = llm.chat(messages)
        except llm_client.SetupProblem as problem:
            print(f"{problem}\n")    # a mistyped /model should not end the chat
            continue
        call_number += 1
        print(f"LLM: {reply.text}\n")
        show_request(call_number, messages, reply)


if __name__ == "__main__":
    try:
        run_chat()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")
