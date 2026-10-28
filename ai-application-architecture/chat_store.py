"""Demo: conversation store — sessions survive restarts, stored in SQLite."""

import argparse
import json

import llm_client
from cli_utils import read_questions, run_safely
from loda.adapters.sqlite_store import DEFAULT_DATABASE, ConversationStore
from loda.features.retrieval import SYSTEM_PROMPT, TOOLS
from loda.pipeline.agent import answer_with_memory, run_agent_loop

DEMO_LINES = [
    "Hi, I'm Priya. I can't find my invoice 1234.",
    "How long will it be kept?",
]


def chat_with_memory(llm: llm_client.Session, store: ConversationStore, session_id: str):
    stored = len(store.load(session_id))
    print(f"LoDA chatbot with {len(TOOLS)} tools on {llm.provider_name}. "
          f"Session '{session_id}' has {stored} stored messages. Type 'exit' to stop.\n")
    for question in read_questions(llm):
        print(f"LLM: {answer_with_memory(llm, store, session_id, question, tools=TOOLS, system=SYSTEM_PROMPT)}\n")


def chat_without_memory(llm: llm_client.Session):
    print(f"LoDA chatbot with {len(TOOLS)} tools, no history: every question is sent alone.\n")
    for question in read_questions(llm):
        print(f"LLM: {run_agent_loop(llm, [{'role': 'user', 'content': question}], TOOLS, SYSTEM_PROMPT)}\n")


def dump_session(session_id: str):
    messages = ConversationStore().load(session_id)
    print(f"Session '{session_id}': {len(messages)} stored message(s) in {DEFAULT_DATABASE.name}\n")
    print(json.dumps(messages, indent=2, ensure_ascii=False))


def run_chat():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session", default="priya")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--forget", action="store_true")
    parser.add_argument("--dump", action="store_true")
    flags, _ = parser.parse_known_args()
    if flags.dump:
        dump_session(flags.session)
        return
    llm = llm_client.Session("demo_conversation", DEMO_LINES)
    if flags.no_history:
        chat_without_memory(llm)
        return
    store = ConversationStore()
    if flags.forget:
        store.forget(flags.session)
    chat_with_memory(llm, store, flags.session)


if __name__ == "__main__":
    run_safely(run_chat)
