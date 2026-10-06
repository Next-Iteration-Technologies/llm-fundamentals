"""
Step 3: retrieval with a sample file.

The tools from step 2 find facts in the archive, but not what the team knows:
the fix for "file too large" lives in data/known_issues.json, not in CSP.
Retrieval finds the few entries that match the question and hands only those
to the model, with their source. No match, no answer: the model hands over.

This is the simplest retrieval there is: count the words a question shares
with each entry. The RAG session replaces it with chunking and embeddings.
The agent loop and the five tools come unchanged from step 2.

Try it:
    uv run step3_retrieval.py --demo
    uv run step3_retrieval.py --replay

    You: My upload failed, it says the file is too large.
    You: I uploaded 20 invoices yesterday. Did they all get archived?
    You: The archive shows a blue screen when I log in.
"""

import re

import archive
import llm_client
from step1_bare_model_call import run_safely
from step2_agent_loop import ARCHIVE_TOOLS, Tool, arguments_schema, chat_in_memory, text_field
from step2_agent_loop import SYSTEM_PROMPT as STEP2_PROMPT

SYSTEM_PROMPT = STEP2_PROMPT + (
    " For error messages and how-to questions, search the known issues first"
    " and name the issue id as your source."
)
DEMO_LINES = [
    "My upload failed, it says the file is too large.",
    "I uploaded 20 invoices yesterday. Did they all get archived?",
    "The archive shows a blue screen when I log in.",
]
MIN_SHARED_WORDS = 2     # fewer shared words than this is not a match
MAX_MATCHES = 2          # only the best entries go to the model, not the whole file
STOP_WORDS = frozenset({
    "and", "the", "for", "but", "not", "you", "are", "was", "with", "this", "that", "what", "why",
    "how", "does", "did", "has", "have", "then", "they", "our", "its", "from", "into", "says", "when",
})


def significant_words(text: str) -> set[str]:
    """The words that carry meaning: lower case, at least three letters, no stop words."""
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2} - STOP_WORDS


def search_known_issues(question: str) -> dict:
    """The known issues that share the most words with the question, best first."""
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
TOOLS = [*ARCHIVE_TOOLS, KNOWLEDGE_TOOL]


def run_chat():
    llm = llm_client.Session("step3_retrieval", DEMO_LINES)
    chat_in_memory(llm, TOOLS, SYSTEM_PROMPT)


if __name__ == "__main__":
    run_safely(run_chat)
