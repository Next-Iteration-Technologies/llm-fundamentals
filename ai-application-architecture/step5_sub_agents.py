"""
Step 5 (optional): sub-agents and a coordinator.

With six tools one agent is fine. As the tool list grows, one agent starts
picking the wrong tool. The pattern: a coordinator that writes a short brief
for a specialist, and specialists that each have only their own tools and a
clean context. To the coordinator, a specialist is just another tool: it
takes a task and returns a short finding.

The cost is real: every specialist is its own agent loop with its own calls.
Run the same questions with --single and compare the number of calls.

Try it:
    uv run step5_sub_agents.py --demo              # the coordinator and two specialists
    uv run step5_sub_agents.py --demo --single     # one agent with all six tools

    You: Find invoices 1234 and 5678, and request access to any I can't open. I'm Priya.
"""

import argparse

import llm_client
from step1_bare_model_call import SYSTEM_PROMPT as STEP1_PROMPT
from step1_bare_model_call import run_safely
from step2_agent_loop import Tool, arguments_schema, run_agent_loop, text_field
from step3_retrieval import SYSTEM_PROMPT as SINGLE_AGENT_PROMPT
from step3_retrieval import TOOLS
from step4_conversation_store import ConversationStore, chat_with_memory

DEMO_LINES = ["Find invoices 1234 and 5678, and request access to any I can't open. I'm Priya."]
COORDINATOR_PROMPT = STEP1_PROMPT + (
    " You coordinate two specialists. Give each one a short brief: the goal, the document ids,"
    " and what to report back in at most three sentences. Combine their findings into one answer."
    " Hand anything they cannot solve to the developers."
)
SPECIALIST_PROMPT = (
    "You are a specialist of the LoDA chatbot. Do only the task in the brief, using your tools."
    " Report back in at most three sentences, with the facts the coordinator needs."
)


def tools_named(tools: list[Tool], *names: str) -> list[Tool]:
    return [tool for tool in tools if tool.name in names]


def sub_agent(llm: llm_client.Session, name: str, description: str, tools: list[Tool]) -> Tool:
    """A specialist, wrapped as a tool: it runs its own agent loop with a clean context."""
    def delegate(task: str) -> str:
        print(f"  [{name}] starts with a clean context and {len(tools)} tools")
        finding = run_agent_loop(llm, [{"role": "user", "content": task}], tools, SPECIALIST_PROMPT)
        print(f"  [{name}] reports back")
        return finding

    parameters = arguments_schema({"task": text_field("The brief: goal, document ids, what to report back")})
    return Tool(name, description, parameters, delegate)


def coordinator_tools(llm: llm_client.Session) -> list[Tool]:
    return [
        sub_agent(llm, "document_agent",
                  "Finds documents, uploads and retention information, and searches known issues.",
                  tools_named(TOOLS, "search_document", "check_upload_status", "get_retention_info",
                              "search_known_issues")),
        sub_agent(llm, "access_agent",
                  "Checks documents and raises access requests to the workspace owner.",
                  tools_named(TOOLS, "search_document", "raise_access_request")),
        *tools_named(TOOLS, "handoff_to_developers"),
    ]


def single_agent_wanted() -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--single", action="store_true")
    flags, _ = parser.parse_known_args()
    return flags.single


def run_chat():
    llm = llm_client.Session("step5_sub_agents", DEMO_LINES)
    store = ConversationStore()
    if single_agent_wanted():
        chat_with_memory(llm, store, "priya-single-agent", TOOLS, SINGLE_AGENT_PROMPT)
    else:
        chat_with_memory(llm, store, "priya-coordinator", coordinator_tools(llm), COORDINATOR_PROMPT)


if __name__ == "__main__":
    run_safely(run_chat)
