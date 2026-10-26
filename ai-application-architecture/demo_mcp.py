"""Demo: MCP client — the chatbot gets its tools from a running MCP server."""

import sys
from pathlib import Path

from mcp import StdioServerParameters

import llm_client
from demo_utils import read_questions, run_safely
from loda.adapters.mcp_tools import McpConnection, tools_from_server
from loda.adapters.sqlite_store import ConversationStore
from loda.features.retrieval import SYSTEM_PROMPT
from loda.pipeline.agent import answer_with_memory

DEMO_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / "demo_mcp_server.py")],
)
DEMO_LINES = [
    "Hi, I'm Priya. I can't find my invoice 1234.",
    "My upload failed, it says the file is too large.",
]


def run_chat():
    llm = llm_client.Session("demo_mcp", DEMO_LINES)
    with McpConnection(DEMO_SERVER) as connection:
        tools = tools_from_server(connection)
        print(f"Server offers: {', '.join(t.name for t in tools)}\n")
        store = ConversationStore()
        for question in read_questions(llm):
            print(f"LLM: {answer_with_memory(llm, store, 'priya-mcp', question, tools=tools, system=SYSTEM_PROMPT)}\n")


if __name__ == "__main__":
    run_safely(run_chat)
