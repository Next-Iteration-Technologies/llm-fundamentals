"""
Step 4: the conversation store.

The model remembers nothing: every call stands alone. A chat only feels like it
remembers because the program sends the whole conversation again with every
question. In steps 2 and 3 that conversation lived in a list and was gone when
the program stopped. Here every message is a row in a SQLite file, so a session
survives a restart. Nothing about it is AI: it is a database table.

Try it:
    uv run step4_conversation_store.py --demo                 # two questions, session "priya"
    uv run step4_conversation_store.py                        # run again and ask:
        You: Can you request access to it for me?             # it still knows "it" is invoice 1234
    uv run step4_conversation_store.py --no-history --demo    # every question sent alone
    uv run step4_conversation_store.py --forget               # start the session from scratch

The tools and the agent loop come unchanged from steps 2 and 3.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import llm_client
from step1_bare_model_call import read_questions, run_safely
from step2_agent_loop import Tool, run_agent_loop
from step3_retrieval import SYSTEM_PROMPT, TOOLS

DEFAULT_DATABASE = Path(__file__).parent / "conversations.db"
DEMO_LINES = [
    "Hi, I'm Priya. I can't find my invoice 1234.",
    "How long will it be kept?",
]


class ConversationStore:
    """Every message of every session: one row each, in the order they were sent."""

    def __init__(self, path: Path | str = DEFAULT_DATABASE):
        self._connection = sqlite3.connect(path)
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                " session_id TEXT, position INTEGER, message TEXT,"
                " PRIMARY KEY (session_id, position))"
            )

    def load(self, session_id: str) -> list[dict]:
        rows = self._connection.execute(
            "SELECT message FROM messages WHERE session_id = ? ORDER BY position", (session_id,)
        )
        return [json.loads(message) for (message,) in rows]

    def save(self, session_id: str, history: list[dict]):
        """Store the messages that are not stored yet. Stored messages never change."""
        stored = len(self.load(session_id))
        with self._connection:
            self._connection.executemany(
                "INSERT INTO messages (session_id, position, message) VALUES (?, ?, ?)",
                [(session_id, position, json.dumps(message, ensure_ascii=False))
                 for position, message in enumerate(history[stored:], start=stored)],
            )

    def forget(self, session_id: str):
        with self._connection:
            self._connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def answer_with_memory(llm: llm_client.Session, store: ConversationStore, session_id: str, question: str,
                       *, tools: list[Tool], system: str, model=None) -> str:
    """THE MEMORY: load the session, add the question, run the loop, store everything new."""
    history = store.load(session_id)
    history.append({"role": "user", "content": question})
    answer = run_agent_loop(llm, history, tools, system, model)
    store.save(session_id, history)
    return answer


def chat_with_memory(llm: llm_client.Session, store: ConversationStore, session_id: str,
                     *, tools: list[Tool], system: str, model=None):
    stored = len(store.load(session_id))
    print(f"LoDA chatbot with {len(tools)} tools on {llm.provider_name}. "
          f"Session '{session_id}' has {stored} stored messages. Type 'exit' to stop.\n")
    for question in read_questions(llm):
        print(f"LLM: {answer_with_memory(llm, store, session_id, question, tools=tools, system=system, model=model)}\n")


def chat_without_memory(llm: llm_client.Session, tools: list[Tool], system: str):
    print(f"LoDA chatbot with {len(tools)} tools, no history: every question is sent alone.\n")
    for question in read_questions(llm):
        print(f"LLM: {run_agent_loop(llm, [{'role': 'user', 'content': question}], tools, system)}\n")


def memory_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session", default="priya")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--forget", action="store_true")
    flags, _ = parser.parse_known_args()
    return flags


def run_chat():
    flags = memory_flags()
    llm = llm_client.Session("step4_conversation_store", DEMO_LINES)
    if flags.no_history:
        chat_without_memory(llm, TOOLS, SYSTEM_PROMPT)
        return
    store = ConversationStore()
    if flags.forget:
        store.forget(flags.session)
    chat_with_memory(llm, store, flags.session, tools=TOOLS, system=SYSTEM_PROMPT)


if __name__ == "__main__":
    run_safely(run_chat)
