"""Demo: agent loop — the model calls tools, we run them, repeat until it answers."""

import llm_client
from cli_utils import read_questions, run_safely
from loda.features.retrieval import SYSTEM_PROMPT, TOOLS
from loda.pipeline.agent import run_agent_loop

DEMO_LINES = [
    "Hi, I'm Priya. I can't find my invoice 1234.",
    "And invoice 5678?",
    "I uploaded 20 invoices yesterday. Did they all get archived?",
    "How long will invoice 1234 be kept?",
]


def run_chat():
    llm = llm_client.Session("demo_agent", DEMO_LINES)
    print(f"LoDA chatbot with {len(TOOLS)} tools on {llm.provider_name}. Type 'exit' to stop.\n")
    history = []
    for question in read_questions(llm):
        history.append({"role": "user", "content": question})
        print(f"LLM: {run_agent_loop(llm, history, TOOLS, SYSTEM_PROMPT)}\n")


if __name__ == "__main__":
    run_safely(run_chat)
