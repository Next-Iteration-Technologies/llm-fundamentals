"""
Round 7: prompt chaining.

One messy ticket, two calls to the model, each with one job.
Call 1 reads the ticket and pulls out the facts as JSON: who, which documents,
what is wrong, how urgent. Call 2 drafts the reply from that JSON only. It
never sees the original ticket. Our code sits between the two calls: it checks
that call 1 gave valid JSON before anything goes further.

Try it:
    uv run r7_chaining.py --demo                       # Nexus, the default
    uv run r7_chaining.py --demo --provider anthropic
    uv run r7_chaining.py --replay                     # the trainer's recorded run

    Ticket: hi, finance here, the invoices from march are missing since the move?? ...
    Ticket: HR again. cant open the old personnel files in the archive ...
    Ticket: pls check - contracts 2019 sales dept, search gives 0 results ...

Without --demo, paste a ticket of your own on one line.
"""

import json
import sys

import llm_client
from request_view import show_request

EXTRACT_PROMPT = (  # CALL 1: read the ticket, answer with facts only
    "You read support tickets for the document archive team. "
    "Answer with one JSON object and nothing else, with exactly these keys:\n"
    '  "requester": the person\'s name, or null\n'
    '  "department": the department the ticket is from, or null\n'
    '  "documents": which documents the ticket is about, or null\n'
    '  "period": the time period of those documents, or null\n'
    '  "problem": what is wrong, in one short sentence\n'
    '  "since": when the problem started, or null\n'
    '  "deadline": a date or day the requester needs it by, or null\n'
    '  "urgency": "high", "normal" or "low"\n'
    "Use null when the ticket does not say. Never guess."
)

DRAFT_PROMPT = (  # CALL 2: write the reply, from the JSON only
    "You draft replies for the document archive team. "
    "You get the facts of a ticket as JSON. Use only those facts. "
    "Rules for the reply: "
    "greet the requester by name, or the department's team if there is no name; "
    "say in one sentence what we understood the problem to be; "
    "say what we will check first; "
    "if there is a deadline, say we have noted it, but never promise a fix date; "
    "for each fact that is null and that we need, ask one short question; "
    "at most 120 words, plain and friendly, no emojis; "
    "sign off as 'Archive Support'."
)

DEMO_LINES = [
    "hi, finance here, the invoices from march are missing since the move?? need them for the audit on friday. thx Sabine",
    "HR again. cant open the old personnel files in the archive, says no permission. worked last week!!! urgent",
    "pls check - contracts 2019 sales dept, search gives 0 results but they were there b4 migration. not urgent but annoying. Tom (Sales Ops)",
]
EXIT_WORDS = {"exit", "quit", "bye"}


def json_from(text):
    """The JSON object in the model's answer, or None. Models sometimes wrap it in ``` fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def run_chain():
    llm = llm_client.Session("r7_chaining", DEMO_LINES, default_provider="nexus", replay_provider="anthropic")
    print(f"Prompt chaining on {llm.provider_name}: ticket -> call 1 (extract JSON) -> call 2 (draft reply).")
    print("Type 'exit' to stop.\n")

    call_number = 0

    while True:
        ticket = llm.read_question("Ticket: ")

        if ticket is None or ticket.lower() in EXIT_WORDS:
            print(f"Goodbye! That was {call_number} calls.")
            break
        if not ticket:
            continue

        # CALL 1: the ticket goes in, JSON comes out
        call_number += 1
        extract_messages = [{"role": "user", "content": ticket}]
        extracted = llm.chat(extract_messages, system=EXTRACT_PROMPT)
        show_request(call_number, extract_messages, extracted, system=EXTRACT_PROMPT)

        # OUR CODE IN THE MIDDLE: only valid JSON goes on to call 2
        facts = json_from(extracted.text)
        if facts is None:
            print(f"Call 1 did not give valid JSON, so we stop here. It said:\n{extracted.text}\n")
            continue
        facts_json = json.dumps(facts, indent=2, ensure_ascii=False)
        print(f"Call 1 extracted:\n{facts_json}\n")

        # CALL 2: only the JSON goes in, the ticket stays behind
        call_number += 1
        draft_messages = [{"role": "user", "content": facts_json}]
        draft = llm.chat(draft_messages, system=DRAFT_PROMPT)
        print(f"Reply draft:\n{draft.text}\n")
        show_request(call_number, draft_messages, draft, system=DRAFT_PROMPT)
        print("Call 2 never saw the ticket, only the JSON from call 1.\n")


if __name__ == "__main__":
    try:
        run_chain()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")
