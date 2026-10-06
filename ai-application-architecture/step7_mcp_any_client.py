"""
Step 7: one server, many clients.

The LoDA MCP server from step 6 belongs to no single program. This script is
a second client with no model at all: it asks the server which tools it has
and calls one directly. n8n and GitHub Copilot connect the same way (see
clients/ and the README). Add a tool to the server once, and every client
sees it at its next start, without a code change.

Try it:
    uv run step7_mcp_any_client.py                                # starts the server over stdio
    uv run step6_mcp_server.py --http                             # in a second terminal, then:
    uv run step7_mcp_any_client.py --url http://127.0.0.1:8000/mcp
"""

import argparse

from step1_bare_model_call import run_safely
from step6_mcp_client import STEP6_SERVER, McpConnection


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
    parser.add_argument("--url", help="an MCP server running over HTTP, for example http://127.0.0.1:8000/mcp")
    flags = parser.parse_args()
    with McpConnection(flags.url or STEP6_SERVER) as connection:
        show_server(connection)


if __name__ == "__main__":
    run_safely(run_client)
