"""
Pattern 05 · Event-Driven Agents

DAS scenario (matches the updated slide): an invoice is archived. Several
things must happen - none depends on the others. Each agent subscribes to
the same event and reacts independently; one failing doesn't block the rest.

    invoice.archived
        -> Index Agent    (adds to the search index)
        -> Notifier Agent (emails the Finance owner)
        -> Audit Agent    (writes a compliance log entry)

Run it:
    uv run labs/05_event_driven.py
"""

from collections import defaultdict
from typing import Callable

import shared


class EventBus:
    """Agents subscribe to events, not to each other. Fully decoupled."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable[[dict], None]]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Callable[[dict], None]) -> None:
        self._subscribers[event_name].append(handler)

    def publish(self, event_name: str, payload: dict) -> None:
        shared.step("Event Bus", f"{event_name} -> {len(self._subscribers[event_name])} subscriber(s)")
        for handler in self._subscribers[event_name]:
            try:
                handler(payload)
            except Exception as error:
                shared.step(handler.__name__, f"FAILED ({error}) - other subscribers still ran")


def index_agent(event: dict) -> None:
    shared.step("Index Agent", f"indexing doc_id={event['doc_id']} for search")


def notifier_agent(event: dict) -> None:
    shared.step("Notifier Agent", f"emailing owner about doc_id={event['doc_id']}")
    note = shared.ask(
        "Write one short email subject line telling a document owner their "
        "invoice has been archived.",
        str(event),
    )
    shared.step("Notifier Agent", f"subject: {note}")


def audit_agent(event: dict) -> None:
    shared.step("Audit Agent", f"logging compliance entry for doc_id={event['doc_id']}")


def main() -> None:
    bus = EventBus()
    bus.subscribe("invoice.archived", index_agent)
    bus.subscribe("invoice.archived", notifier_agent)
    bus.subscribe("invoice.archived", audit_agent)

    bus.publish("invoice.archived", {"doc_id": "1234", "title": "Invoice 1234, Meyer Logistics"})


if __name__ == "__main__":
    main()
