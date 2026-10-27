"""
Pattern 01 · Sequential Pipeline

DAS scenario (matches the slide): Priya types "I can't find invoice 1234."
The query flows through a fixed pipeline, each stage handing off to the next:

    Intent Parser -> Query Builder -> Retriever -> Formatter

Run it:
    uv run labs/01_sequential_pipeline.py
"""

import shared


def parse_intent(query: str) -> dict:
    shared.step("Intent Parser", f"query={query!r}")
    reply = shared.ask(
        "Extract the document type and id from a Finance user's question. "
        "Answer with exactly two words: the doc type, then the id. "
        "Example: 'invoice 1234'.",
        query,
    )
    doc_type, doc_id = reply.strip().split()[:2]
    return {"doc_type": doc_type, "doc_id": doc_id}


def build_query(intent: dict) -> dict:
    shared.step("Query Builder", f"intent={intent}")
    return {"doc_id": intent["doc_id"]}


def retrieve(query: dict) -> dict | None:
    shared.step("Retriever", f"query={query}")
    return shared.lookup_invoice(query["doc_id"])


def format_reply(record: dict | None) -> str:
    shared.step("Formatter", f"record={record}")
    if record is None:
        return shared.ask(
            "Write one short, friendly sentence telling a Finance user the "
            "document was not found in the archive.",
            "not found",
        )
    return shared.ask(
        "Write one short, friendly sentence for a Finance user summarising "
        "this archived document record.",
        str(record),
    )


def main() -> None:
    query = "I can't find invoice 1234."
    intent = parse_intent(query)
    built = build_query(intent)
    record = retrieve(built)
    answer = format_reply(record)
    print(f"\nPriya sees: {answer}")


if __name__ == "__main__":
    main()
