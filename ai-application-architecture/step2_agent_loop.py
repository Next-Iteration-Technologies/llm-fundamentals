"""
Step 2: the agent loop and tools.

The model guessed in step 1 because it cannot see the archive. So we give it
tools. A tool is a plain Python function on this laptop, plus a name, a
description and parameters that the model reads. The model never runs
anything: it answers "please run search_document(doc_id='1234')", our code
runs it, and the result goes back in the next call. We repeat until the model
answers in words. That repetition is the agent loop.

Try it:
    uv run step2_agent_loop.py --demo
    uv run step2_agent_loop.py --replay

    You: Hi, I'm Priya. I can't find my invoice 1234.
    You: And invoice 5678?
    You: I uploaded 20 invoices yesterday. Did they all get archived?
    You: How long will invoice 1234 be kept?

The conversation is kept in a list while the program runs. Step 4 stores it.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

import archive
import llm_client
from request_view import call_text, shorten
from step1_bare_model_call import SYSTEM_PROMPT as STEP1_PROMPT
from step1_bare_model_call import read_questions, run_safely, show_call

SYSTEM_PROMPT = STEP1_PROMPT + (
    " Use the tools to check the archive and answer only from their results."
    " If the tools cannot answer, hand the question to the developers with a short summary."
)
DEMO_LINES = [
    "Hi, I'm Priya. I can't find my invoice 1234.",
    "And invoice 5678?",
    "I uploaded 20 invoices yesterday. Did they all get archived?",
    "How long will invoice 1234 be kept?",
]
MAX_CALLS_PER_QUESTION = 6       # the brake: stop if the model keeps asking for tools
STEP_LIMIT_ANSWER = "I could not finish this within my step limit. Please contact the LoDA developers."


@dataclass(frozen=True)
class Tool:
    """A plain Python function, plus what the model reads about it."""
    name: str
    description: str
    parameters: dict              # a JSON schema of the arguments
    function: Callable[..., object]

    def definition(self) -> dict:
        """What goes to the model with every call: everything except the function."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def text_field(description: str) -> dict:
    return {"type": "string", "description": description}


def arguments_schema(required: dict, optional: dict | None = None) -> dict:
    return {"type": "object", "properties": {**required, **(optional or {})}, "required": list(required)}


# THE TOOLS: the descriptions are prompts. The model picks a tool by reading them.
ARCHIVE_TOOLS = [
    Tool(
        name="search_document",
        description="Find an archived document by its id. Returns whether it exists, its title, "
                    "workspace, folder and a link. Read-only. Use it when someone asks where a "
                    "document is or cannot find it.",
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
                    "and why. Read-only. Use it when someone asks whether an upload worked.",
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


def as_text(result: object) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def call_safely(function: Callable[..., object], arguments: dict) -> str:
    """Run a tool. Arguments the function doesn't take go back to the model as an error, not a crash."""
    try:
        return as_text(function(**arguments))
    except TypeError as error:
        return as_text({"error": f"Wrong arguments: {error}"})


def run_tool_call(llm: llm_client.Session, call: dict, tools_by_name: dict[str, Tool]) -> dict:
    """Run one tool the model asked for, on this laptop, and turn the result into a message."""
    tool = tools_by_name.get(call["name"])
    if tool is None:
        result = as_text({"error": f"There is no tool called {call['name']}."})
    else:
        result = llm.run_tool(call, lambda **arguments: call_safely(tool.function, arguments))
    print(f"  ran {call_text(call)} -> {shorten(result)}")
    return {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result}


def run_agent_loop(llm: llm_client.Session, history: list[dict], tools: list[Tool], system: str,
                   model=None) -> str:
    """THE LOOP: call the model; while it asks for tools, run them and call again.

    history grows in place: the question, every tool call and result, then the answer.
    model is anything with .chat(); later steps put a gateway or a redactor there.
    """
    model = model or llm
    definitions = [tool.definition() for tool in tools]
    tools_by_name = {tool.name: tool for tool in tools}

    for _ in range(MAX_CALLS_PER_QUESTION):
        reply = model.chat(history, system=system, tools=definitions)
        show_call(history, reply, system=system, tools=definitions)
        history.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
        if not reply.tool_calls:
            return reply.text
        for call in reply.tool_calls:
            history.append(run_tool_call(llm, call, tools_by_name))

    history.append({"role": "assistant", "content": STEP_LIMIT_ANSWER})
    return STEP_LIMIT_ANSWER


def chat_in_memory(llm: llm_client.Session, tools: list[Tool], system: str):
    """A chat whose history lives in a list: it is gone when the program stops."""
    print(f"LoDA chatbot with {len(tools)} tools on {llm.provider_name}. Type 'exit' to stop.\n")
    history = []
    for question in read_questions(llm):
        history.append({"role": "user", "content": question})
        print(f"LLM: {run_agent_loop(llm, history, tools, system)}\n")


def run_chat():
    llm = llm_client.Session("step2_agent_loop", DEMO_LINES)
    chat_in_memory(llm, ARCHIVE_TOOLS, SYSTEM_PROMPT)


if __name__ == "__main__":
    run_safely(run_chat)
