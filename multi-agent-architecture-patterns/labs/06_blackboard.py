"""
Pattern 06 · Blackboard / Peer-to-Peer

DAS scenario (matches the slide): Priya asks "Why was invoice 1234 rejected
last quarter?" - investigative, no fixed answer path. Agents read and write
a shared workspace; no one is "in charge". Synthesis reads everything and
drafts the answer.

    Blackboard (shared dict)
        Retrieval Agent -> posts raw invoice data
        Policy Agent    -> posts relevant rejection rules
        Audit Agent     -> posts matching audit log entries
        Synthesis Agent -> reads all, drafts the explanation

Run it:
    uv run labs/06_blackboard.py
"""

import shared


def retrieval_agent(blackboard: dict, doc_id: str) -> None:
    shared.step("Retrieval Agent", f"posting invoice data for {doc_id!r}")
    blackboard["invoice"] = shared.lookup_invoice(doc_id)


def policy_agent(blackboard: dict) -> None:
    shared.step("Policy Agent", "reading invoice, posting rejection rules")
    blackboard["policy_notes"] = shared.ask(
        "Given this archived invoice record, name one plausible Finance "
        "policy reason an invoice like this could have been rejected "
        "(e.g. missing PO, duplicate, over budget). One short sentence.",
        str(blackboard["invoice"]),
    )


def audit_agent(blackboard: dict) -> None:
    shared.step("Audit Agent", "reading blackboard, posting matching log entries")
    blackboard["audit_notes"] = shared.ask(
        "Given this invoice record and policy note, write one short "
        "sentence describing what an audit log entry for its rejection "
        "might say.",
        f"Invoice: {blackboard['invoice']}\nPolicy: {blackboard['policy_notes']}",
    )


def synthesis_agent(blackboard: dict) -> str:
    shared.step("Synthesis Agent", "reading all contributions, drafting explanation")
    return shared.ask(
        "Using everything posted to this shared workspace, write a short, "
        "clear explanation for a Finance user of why the invoice was "
        "likely rejected.",
        str(blackboard),
    )


def main() -> None:
    blackboard: dict = {}
    shared.step("Blackboard", "initialised with query context")

    retrieval_agent(blackboard, doc_id="1234")
    policy_agent(blackboard)
    audit_agent(blackboard)
    answer = synthesis_agent(blackboard)

    print(f"\nPriya sees: {answer}")


if __name__ == "__main__":
    main()
