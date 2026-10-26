"""Demo: a bare model call — no tools, no memory."""

import llm_client
from demo_utils import read_questions, run_safely, show_call

SYSTEM_PROMPT = (
    "You are the LoDA chatbot. You help department users with questions about the LoDA "
    "document archive (CSP and Classic). Answer briefly and in plain language."
)
DEMO_LINES = [
    "Hi, I'm Priya. I can't find my invoice 1234.",
    "Is it in the 2024 folder?",
]


def ask_model(llm: llm_client.Session, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    reply = llm.chat(messages, system=SYSTEM_PROMPT)
    show_call(messages, reply, system=SYSTEM_PROMPT)
    return reply.text


def run_chat():
    llm = llm_client.Session("demo_bare", DEMO_LINES)
    print(f"LoDA chatbot, bare model call on {llm.provider_name}. Type 'exit' to stop.\n")
    for question in read_questions(llm):
        print(f"LLM: {ask_model(llm, question)}\n")


if __name__ == "__main__":
    run_safely(run_chat)
