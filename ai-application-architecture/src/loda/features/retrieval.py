"""Retrieval feature: keyword search over known issues + full tool list."""

import re

import archive
from loda.core.tools import Tool, arguments_schema, text_field

_BASE_SYSTEM_PROMPT = (
    "You are the LoDA chatbot. You help department users with the LoDA archive:"
    " finding documents, checking uploads, retention and access."
    " Use the tools to check the archive and answer only from their results."
    " If the tools cannot answer, hand the question to the developers with a short summary."
    " For error messages and how-to questions, search the known issues first"
    " and name the issue id as your source."
)

SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT

MIN_SHARED_WORDS = 2
MAX_MATCHES = 2
_STOP_WORDS = frozenset({
    "and", "the", "for", "but", "not", "you", "are", "was", "with", "this", "that", "what", "why",
    "how", "does", "did", "has", "have", "then", "they", "our", "its", "from", "into", "says", "when",
})


def significant_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2} - _STOP_WORDS


def search_known_issues(question: str) -> dict:
    asked = significant_words(question)
    scored = [
        (len(asked & significant_words(f"{issue['symptom']} {issue['fix']}")), issue)
        for issue in archive.read_data("known_issues.json")
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    matches = [issue for shared, issue in scored if shared >= MIN_SHARED_WORDS][:MAX_MATCHES]
    if not matches:
        return {"matches": [], "note": "No known issue matches. Do not guess: hand over to the developers."}
    return {"matches": matches}


KNOWLEDGE_TOOL = Tool(
    name="search_known_issues",
    description="Search the team's known issues and how-tos for an error message or question. "
                "Returns the matching entries with their fix and source. Read-only.",
    parameters=arguments_schema({"question": text_field("The error message or question, in the user's words")}),
    function=search_known_issues,
)

ARCHIVE_TOOLS = [
    Tool(
        name="search_document",
        description="Find an archived document by its id. Returns whether it exists, its title, "
                    "workspace, folder and a link. Read-only.",
        parameters=arguments_schema({"doc_id": text_field("The document id, digits only, for example 1234")}),
        function=archive.search_document,
    ),
    Tool(
        name="raise_access_request",
        description="Ask the workspace owner to give someone access to a document. Creates a request "
                    "in ServiceNow; a person decides. Use it when someone cannot open a document.",
        parameters=arguments_schema({
            "doc_id": text_field("The document id"),
            "reason": text_field("Why access is needed, in the user's words"),
            "requested_by": text_field("Who needs access"),
        }),
        function=archive.raise_access_request,
    ),
    Tool(
        name="check_upload_status",
        description="Show the status of recent uploads: how many files were archived, which failed "
                    "and why. Read-only.",
        parameters=arguments_schema({}, {
            "uploaded_by": text_field("Who uploaded the files"),
            "upload_id": text_field("The upload id, if the user knows it, for example UP-2041"),
        }),
        function=archive.check_upload_status,
    ),
    Tool(
        name="get_retention_info",
        description="How long a document is kept: its retention class, archive date and planned "
                    "deletion date. Read-only.",
        parameters=arguments_schema({"doc_id": text_field("The document id")}),
        function=archive.get_retention_info,
    ),
    Tool(
        name="handoff_to_developers",
        description="Open a ticket for the LoDA developer team. Use it for anything the other tools "
                    "cannot answer, such as missing documents after a migration or a system that is down.",
        parameters=arguments_schema(
            {"summary": text_field("Who asked, what they asked, what you already checked, and the document ids")},
            {"urgency": {"type": "string", "enum": ["normal", "urgent"]}},
        ),
        function=archive.handoff_to_developers,
    ),
]

TOOLS = [*ARCHIVE_TOOLS, KNOWLEDGE_TOOL]
