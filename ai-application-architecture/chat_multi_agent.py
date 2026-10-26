"""Demo: sub-agents — a coordinator delegates to specialist agents."""

import argparse

import llm_client
from cli_utils import read_questions, run_safely
from loda.adapters.sqlite_store import ConversationStore
from loda.core.tools import Tool, arguments_schema, text_field
from loda.features.retrieval import SYSTEM_PROMPT as RETRIEVAL_PROMPT
from loda.features.retrieval import TOOLS
from loda.pipeline.agent import answer_with_memory, run_agent_loop

DEMO_LINES = ["Find invoices 1234 and 5678, and request access to any I can't open. I'm Priya."]
COORDINATOR_PROMPT = (
    "You are the LoDA chatbot. You help department users with questions about the LoDA document archive."
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
    def delegate(task: str) -> str:
        print(f"  [{name}] starts with a clean context and {len(tools)} tools")
        finding = run_agent_loop(llm, [{"role": "user", "content": task}], tools, SPECIALIST_PROMPT)
        print(f"  [{name}] reports back")
        return finding

    return Tool(name, description, arguments_schema({"task": text_field("The brief: goal, document ids, what to report back")}), delegate)


def coordinator_tools(llm: llm_client.Session) -> list[Tool]:
    return [
        sub_agent(llm, "document_agent",
                  "Finds documents, uploads and retention information, and searches known issues.",
                  tools_named(TOOLS, "search_document", "check_upload_status", "get_retention_info", "search_known_issues")),
        sub_agent(llm, "access_agent",
                  "Checks documents and raises access requests to the workspace owner.",
                  tools_named(TOOLS, "search_document", "raise_access_request")),
        *tools_named(TOOLS, "handoff_to_developers"),
    ]


def run_chat():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--single", action="store_true")
    flags, _ = parser.parse_known_args()
    llm = llm_client.Session("demo_sub_agents", DEMO_LINES)
    store = ConversationStore()
    if flags.single:
        session = "priya-single-agent"
        stored = len(store.load(session))
        print(f"Single agent, {len(TOOLS)} tools. Session '{session}' has {stored} messages.\n")
        for question in read_questions(llm):
            print(f"LLM: {answer_with_memory(llm, store, session, question, tools=TOOLS, system=RETRIEVAL_PROMPT)}\n")
    else:
        tools = coordinator_tools(llm)
        session = "priya-coordinator"
        stored = len(store.load(session))
        print(f"Coordinator + specialists. Session '{session}' has {stored} messages.\n")
        for question in read_questions(llm):
            print(f"LLM: {answer_with_memory(llm, store, session, question, tools=tools, system=COORDINATOR_PROMPT)}\n")


if __name__ == "__main__":
    run_safely(run_chat)
