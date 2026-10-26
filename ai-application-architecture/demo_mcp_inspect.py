"""Demo: MCP inspector — list tools and call one directly, no model."""

import argparse
import sys
from pathlib import Path

from mcp import StdioServerParameters

from demo_utils import run_safely
from loda.adapters.mcp_tools import McpConnection

DEMO_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / "demo_mcp_server.py")],
)


def first_sentence(text: str) -> str:
    return text.split(". ", maxsplit=1)[0].rstrip(".") + "."


def show_server(connection: McpConnection):
    print("Tools on the LoDA MCP server:")
    for tool in connection.list_tools():
        print(f"  {tool.name:<22} {first_sentence(tool.description or '')}")
    print("\nsearch_document(doc_id='1234') returned:")
    print(connection.call_tool("search_document", {"doc_id": "1234"}))


def run_client():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="an MCP server running over HTTP, e.g. http://127.0.0.1:8000/mcp")
    flags = parser.parse_args()
    with McpConnection(flags.url or DEMO_SERVER) as connection:
        show_server(connection)


if __name__ == "__main__":
    run_safely(run_client)
