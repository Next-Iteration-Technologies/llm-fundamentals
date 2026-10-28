"""
The mock DAS archive, played by data/documents.json.

Nothing here is AI. This is the system every pattern's agents look things up
in. Every lab reads from the same six records, so results stay comparable
across patterns.
"""

import json
from functools import cache
from pathlib import Path

DATA_FOLDER = Path(__file__).parent / "data"


@cache
def read_data(file_name: str) -> list[dict]:
    """One file from data/, read once per run."""
    return json.loads((DATA_FOLDER / file_name).read_text(encoding="utf-8"))


def find_document(doc_id: str) -> dict | None:
    return next((doc for doc in read_data("documents.json") if doc["doc_id"] == doc_id.strip()), None)
