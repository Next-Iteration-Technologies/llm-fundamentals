"""SQLite-backed ConversationStore implementing loda.core.store.ConversationStore."""

import json
import sqlite3
from pathlib import Path

DEFAULT_DATABASE = Path(__file__).parents[4] / "conversations.db"
IN_MEMORY_STORE = ":memory:"


class ConversationStore:
    """Every message of every session persisted in SQLite; one row per message."""

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
            "SELECT message FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        )
        return [json.loads(message) for (message,) in rows]

    def save(self, session_id: str, history: list[dict]) -> None:
        stored = len(self.load(session_id))
        with self._connection:
            self._connection.executemany(
                "INSERT INTO messages (session_id, position, message) VALUES (?, ?, ?)",
                [
                    (session_id, position, json.dumps(message, ensure_ascii=False))
                    for position, message in enumerate(history[stored:], start=stored)
                ],
            )

    def forget(self, session_id: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
