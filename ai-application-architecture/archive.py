"""
The mock LoDA archive: CSP, Classic and ServiceNow, played by the files in data/.

Nothing here is AI. These are the systems the chatbot's tools talk to. In the
training they read data/ instead of production, so every laptop gets the same
answers. Every operation returns a plain dict that can go back to the model.
"""

import json
from functools import cache
from pathlib import Path

DATA_FOLDER = Path(__file__).parent / "data"
ARCHIVE_URL = "https://archive.example.internal/documents"


@cache
def read_data(file_name: str) -> list[dict]:
    """One file from data/, read once per run."""
    return json.loads((DATA_FOLDER / file_name).read_text(encoding="utf-8"))


def find_document(doc_id: str) -> dict | None:
    return next((doc for doc in read_data("documents.json") if doc["doc_id"] == doc_id.strip()), None)


def is_same_person(given_name: str, user_id: str) -> bool:
    """'Priya', 'priya' and 'Priya Sharma' all mean the user 'priya'."""
    words = given_name.strip().lower().split()
    return bool(words) and words[0] == user_id


class ServiceNow:
    """Access requests and tickets, kept in memory for one run of the program."""

    def __init__(self):
        self.records: list[dict] = []

    def create(self, prefix: str, **fields) -> dict:
        record = {"number": f"{prefix}{1001 + len(self.records)}", **fields}
        self.records.append(record)
        return record


SERVICENOW = ServiceNow()


# ---------- the operations behind the chatbot's tools ----------

def search_document(doc_id: str) -> dict:
    """Find an archived document by its id."""
    document = find_document(doc_id)
    if document is None:
        return {"found": False, "doc_id": doc_id}
    return {
        "found": True,
        "doc_id": document["doc_id"],
        "title": document["title"],
        "workspace": document["workspace"],
        "folder": document["folder"],
        "platform": document["platform"],
        "link": f"{ARCHIVE_URL}/{document['doc_id']}",
    }


def raise_access_request(doc_id: str, reason: str, requested_by: str) -> dict:
    """Ask the workspace owner to grant access. A person decides; this only creates the request."""
    document = find_document(doc_id)
    if document is None:
        return {"created": False, "error": f"There is no document {doc_id}."}
    request = SERVICENOW.create(
        "RITM",
        kind="access request",
        doc_id=document["doc_id"],
        requested_by=requested_by,
        reason=reason,
        approver=document["owner"],
        status="Waiting for the workspace owner",
    )
    return {"created": True, **request}


def check_upload_status(uploaded_by: str = "", upload_id: str = "") -> dict:
    """The status of recent uploads, found by who uploaded them or by upload id."""
    uploads = [
        upload for upload in read_data("uploads.json")
        if upload["upload_id"] == upload_id or is_same_person(uploaded_by, upload["uploaded_by"])
    ]
    if not uploads:
        return {"uploads": [], "note": "No upload found."}
    return {"uploads": uploads}


def get_retention_info(doc_id: str) -> dict:
    """How long a document is kept, and when it is planned to be deleted."""
    document = find_document(doc_id)
    if document is None:
        return {"found": False, "doc_id": doc_id}
    return {
        "found": True,
        "doc_id": document["doc_id"],
        "retention_class": document["retention_class"],
        "archived_on": document["archived_on"],
        "delete_after": document["delete_after"],
    }


def handoff_to_developers(summary: str, urgency: str = "normal") -> dict:
    """Open a ticket for the LoDA developer team."""
    ticket = SERVICENOW.create(
        "INC",
        kind="incident",
        assignment_group="LoDA developers",
        summary=summary,
        urgency=urgency,
        status="New",
    )
    return {"created": True, **ticket}
