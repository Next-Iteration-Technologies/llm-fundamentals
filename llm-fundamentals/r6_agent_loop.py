"""
Round 6: the agent loop.

In R4 the model asked for the weather in Berlin and Munich at the same time:
it knew both cities before it called a single tool. Here it doesn't. To answer
"I can't find invoice 4711" it has to search first, and only the search result
tells it which workspace to check next. So the loop goes round several times
in one turn, and each pass decides the next one:

    THINK    the model reads everything so far and picks the next tool (or answers)
    ACT      our code runs that tool
    OBSERVE  our code appends the result to the list, and we call the model again

The archive is MOCKED: a few dictionaries in this file stand in for CSP,
because we have no connection to the real one. The loop doesn't care.

Try it:
    uv run r6_agent_loop.py --demo                       # Nexus, the default
    uv run r6_agent_loop.py --demo --provider anthropic
    uv run r6_agent_loop.py --replay                     # the trainer's recorded run

    You: I'm Priya. I can't find invoice 4711.
    You: Who owns the HR workspace?
    You: I'm Priya. I can't find invoice 9999.
"""

import json
import sys

import llm_client
from request_view import call_text, shorten, show_request

DEMO_LINES = [
    "I'm Priya. I can't find invoice 4711.",
    "Who owns the HR workspace?",
    "I'm Priya. I can't find invoice 9999.",
]
EXIT_WORDS = {"exit", "quit", "bye"}
MAX_CALLS_PER_QUESTION = 6       # stop if the model keeps asking for tools


# ---------- MOCK ARCHIVE: stands in for CSP, nothing here is real ----------

DOCUMENTS = [
    {"id": "INV-4711", "title": "Invoice 4711, Kaufmann Logistik", "workspace": "Finance", "archived": "2024-03-14"},
    {"id": "INV-4712", "title": "Invoice 4712, Kaufmann Logistik", "workspace": "Finance", "archived": "2024-03-14"},
    {"id": "PO-2210", "title": "Purchase order 2210, office chairs", "workspace": "Procurement", "archived": "2025-01-20"},
    {"id": "HR-0815", "title": "Employment contract template 2025", "workspace": "HR", "archived": "2025-02-03"},
]
ACCESS = {
    "Finance": ["Markus Weber", "Elena Schmidt", "Tobias Klein"],
    "Procurement": ["Priya", "Jonas Becker"],
    "HR": ["Sabine Wolf", "Ahmet Yilmaz"],
}
OWNERS = {
    "Finance": {"name": "Markus Weber", "contact": "markus.weber@example.com"},
    "Procurement": {"name": "Jonas Becker", "contact": "jonas.becker@example.com"},
    "HR": {"name": "Sabine Wolf", "contact": "sabine.wolf@example.com"},
}


# ---------- the tools: plain functions over the mock archive ----------

def find_workspace(name):
    return next((workspace for workspace in ACCESS if workspace.lower() == name.strip().lower()), None)


def search_document(query):
    words = query.lower().split()
    found = [doc for doc in DOCUMENTS if all(word in f"{doc['id']} {doc['title']}".lower() for word in words)]
    return json.dumps({"query": query, "found": found})


def user_access_report(workspace):
    name = find_workspace(workspace)
    if name is None:
        return json.dumps({"error": f"No workspace called {workspace}."})
    return json.dumps({"workspace": name, "people_with_access": ACCESS[name]})


def workspace_owner(workspace):
    name = find_workspace(workspace)
    if name is None:
        return json.dumps({"error": f"No workspace called {workspace}."})
    return json.dumps({"workspace": name, "owner": OWNERS[name]})


TOOL_FUNCTIONS = {
    "search_document": search_document,
    "user_access_report": user_access_report,
    "workspace_owner": workspace_owner,
}

# ---------- what the model sees: name, description, parameters ----------

TOOLS = [  # THE TOOLS: sent with every call, like the system prompt
    {
        "name": "search_document",
        "description": "Searches the whole archive, across all workspaces, regardless of who is asking. "
                       "Returns matching documents with their workspace and archive date.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Words to search for, for example 'invoice 4711'"}},
            "required": ["query"],
        },
    },
    {
        "name": "user_access_report",
        "description": "Lists the people who can open documents in one workspace.",
        "parameters": {
            "type": "object",
            "properties": {"workspace": {"type": "string", "description": "Workspace name, for example Finance"}},
            "required": ["workspace"],
        },
    },
    {
        "name": "workspace_owner",
        "description": "Returns the owner of a workspace: the person who can grant access to it.",
        "parameters": {
            "type": "object",
            "properties": {"workspace": {"type": "string", "description": "Workspace name, for example Finance"}},
            "required": ["workspace"],
        },
    },
]


def run_chat():
    llm = llm_client.Session("r6_agent_loop", DEMO_LINES, default_provider="nexus", replay_provider="anthropic")
    offered = ", ".join(tool["name"] for tool in TOOLS)
    print(f"Archive assistant with tools ({offered}) on {llm.provider_name}. Type 'exit' to stop.\n")

    chat_history = []
    call_number = 0

    while True:
        question = llm.read_question()

        if question is None or question.lower() in EXIT_WORDS:
            print(f"Goodbye! We exchanged {len(chat_history)} messages.")
            break
        if not question:
            continue

        chat_history.append({"role": "user", "content": question})
        model_calls = tool_calls = 0

        # THE LOOP: think, act, observe, until the model stops asking for tools
        for pass_number in range(1, MAX_CALLS_PER_QUESTION + 1):
            print(f"── pass {pass_number} " + "─" * 40)
            call_number += 1
            model_calls += 1
            reply = llm.chat(chat_history, tools=TOOLS)                       # THINK
            chat_history.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})

            if not reply.tool_calls:
                print("THINK    (model)    no tool wanted, so it answers")
                show_request(call_number, chat_history[:-1], reply, tools=TOOLS)
                print(f"LLM: {reply.text}\n")
                break

            if reply.text:
                print(f"THINK    (model)    \"{shorten(reply.text)}\"")
            for call in reply.tool_calls:
                print(f"THINK    (model)    wants: {call_text(call)}")
            show_request(call_number, chat_history[:-1], reply, tools=TOOLS)

            for call in reply.tool_calls:
                function = TOOL_FUNCTIONS.get(call["name"], lambda **_: f"There is no tool called {call['name']}.")
                result = llm.run_tool(call, function)                        # ACT
                tool_calls += 1
                where = "in the recording" if llm.replay else "on this laptop"
                print(f"ACT      (our code) ran {call['name']} {where}")
                # OBSERVE: the same append as THE MEMORY in R2; the result is now part of the conversation
                chat_history.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result})
                print(f"OBSERVE  (our code) appended: {result}\n")
        else:
            print(f"(Stopped after {MAX_CALLS_PER_QUESTION} calls for one question.)\n")

        print(f"loop ran {model_calls} times · {model_calls} model calls · {tool_calls} tool calls · 1 turn for you\n")


if __name__ == "__main__":
    try:
        run_chat()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")
