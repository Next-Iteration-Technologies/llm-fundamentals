"""
Round 0: the model continues text.

A base model has never been taught to chat. It does one thing: predict the
next word, over and over. We send it a plain string, with no system, no user
and no assistant, and it simply keeps writing.

Each start is run at temperature 0 and at temperature 1, twice each. The
request is printed once, then one line per run with what the model added.

Try it:
    uv run r0_continue_text.py --demo        # the prepared starts, on Ollama
    uv run r0_continue_text.py --replay      # the trainer's recorded run, no Ollama needed
    uv run r0_continue_text.py               # type your own starts
"""

import sys

import llm_client
from request_view import show_continuations, show_text_request

TEXT_STARTS = [
    "The capital of Germany is Berlin. The capital of France is",
    "What is the capital of Italy?",
]
TEMPERATURES = [0.0, 1.0]
RUNS_PER_TEMPERATURE = 2
MAX_NEW_TOKENS = 40
EXIT_WORDS = {"exit", "quit", "bye"}


def run_continuations():
    llm = llm_client.Session("r0_continue_text", TEXT_STARTS, default_provider="ollama")
    print("Round 0: the model continues text. Type 'exit' to stop.\n")

    while True:
        text = llm.read_question("Start of a text: ")

        if text is None or text.lower() in EXIT_WORDS:
            print("Goodbye!")
            break
        if not text:
            continue

        print(f"(asking the model {len(TEMPERATURES) * RUNS_PER_TEMPERATURE} times...)\n")
        runs = []
        for temperature in TEMPERATURES:
            for run in range(1, RUNS_PER_TEMPERATURE + 1):
                reply = llm.complete(text, temperature, MAX_NEW_TOKENS)
                runs.append((temperature, run, reply.text))

        show_text_request(text, reply)
        show_continuations(runs)


if __name__ == "__main__":
    try:
        run_continuations()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")
