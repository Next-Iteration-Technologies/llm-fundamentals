"""
Pattern 02 · Coordinator + Specialists

DAS scenario (matches the slide): Priya asks "Find invoice 1234 and check if
it's overdue." Two concerns, one message. The coordinator never does the
work itself - it routes to specialists and merges what they return.

    Coordinator -> SQL Specialist (lookup)
                -> Policy Specialist (overdue check)
                -> merges both into one reply

Run it:
    uv run labs/02_coordinator_specialists.py
"""

import shared


def sql_specialist(doc_id: str) -> dict | None:
    shared.step("SQL Specialist", f"doc_id={doc_id}")
    return shared.lookup_invoice(doc_id)


def policy_specialist(record: dict | None) -> str:
    shared.step("Policy Specialist", f"record={record}")
    if record is None:
        return "no record to check"
    return shared.ask(
        "Given this archived invoice record, state in one short sentence "
        "whether it looks overdue, using only the fields given (archived_on, "
        "delete_after). If the record doesn't contain payment-due data, say so plainly.",
        str(record),
    )


def coordinator(query: str, doc_id: str) -> str:
    shared.step("Coordinator", f"query={query!r} -> delegating")
    record = sql_specialist(doc_id)
    policy_note = policy_specialist(record)

    shared.step("Coordinator", "merging specialist results")
    return shared.ask(
        "Merge these two specialist findings into one short, friendly reply "
        "for a Finance user. First finding is the record lookup, second is "
        "the policy check.",
        f"Lookup: {record}\nPolicy: {policy_note}",
    )


def main() -> None:
    query = "Find invoice 1234 and check if it's overdue."
    answer = coordinator(query, doc_id="1234")
    print(f"\nPriya sees: {answer}")


if __name__ == "__main__":
    main()
