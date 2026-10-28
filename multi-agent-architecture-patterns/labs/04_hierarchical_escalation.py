"""
Pattern 04 · Hierarchical Escalation

DAS scenario (matches the slide): Priya mistypes "Find invoice 12334" (an
extra digit). Each tier tries at its own level first; escalation happens
only when the tier below couldn't resolve it, and each tier logs why.

    Tier 1 Dev (basic retrieval)
      -> Tier 2 TL (fuzzy match across legacy + new)
        -> Tier 3 EM (compliance / legal-hold check)

Run it:
    uv run labs/04_hierarchical_escalation.py
"""

import difflib

import archive

import shared


def tier1_basic_retriever(doc_id: str) -> dict | None:
    shared.step("Tier 1 · Dev", f"basic retrieval for {doc_id!r}")
    return shared.lookup_invoice(doc_id)


def tier2_advanced_retriever(doc_id: str) -> dict | None:
    shared.step("Tier 2 · TL", f"escalated: fuzzy match for {doc_id!r}")
    known_ids = [doc["doc_id"] for doc in archive.read_data("documents.json")]
    close = difflib.get_close_matches(doc_id, known_ids, n=1, cutoff=0.6)
    return shared.lookup_invoice(close[0]) if close else None


def tier3_compliance_reviewer(record: dict) -> str:
    shared.step("Tier 3 · EM", f"escalated: compliance check for {record['doc_id']!r}")
    return shared.ask(
        "Given this archived document's retention_class and delete_after "
        "fields, state in one short sentence whether it appears to be under "
        "a legal hold or long-term retention that Finance should be aware of.",
        str(record),
    )


def main() -> None:
    requested_id = "12334"  # Priya's typo for invoice 1234

    record = tier1_basic_retriever(requested_id)
    if record is None:
        record = tier2_advanced_retriever(requested_id)

    if record is None:
        print("\nPriya sees: Not found at any tier.")
        return

    compliance_note = tier3_compliance_reviewer(record)
    answer = shared.ask(
        "Write one short, friendly sentence for a Finance user: it found a "
        "close match for their requested id, summarise the record, and "
        "fold in this compliance note.",
        f"Record: {record}\nCompliance: {compliance_note}",
    )
    print(f"\nPriya sees: {answer}")


if __name__ == "__main__":
    main()
