"""
Step 6, the server: the LoDA MCP server.

Until now the tool code lived inside the chatbot. Any other program that
needed search_document had to copy it. Here the six tools move into one MCP
server. Any MCP client can connect, ask "which tools do you have?" and call
them: the chatbot (step 6, client), a plain script (step 7), n8n, GitHub
Copilot. The server stays stateless: it only runs tools. The conversation
stays with the chatbot.

You rarely start it yourself: the chatbot starts it over stdio. To let other
clients connect over the network, start it with HTTP:
    uv run step6_mcp_server.py --http              # http://127.0.0.1:8000/mcp
"""

import argparse

from mcp.server.mcpserver import MCPServer

from step2_agent_loop import Tool
from step3_retrieval import TOOLS

SERVER_NAME = "loda-archive"


def build_server(tools: list[Tool]) -> MCPServer:
    """One MCP tool per chatbot tool: the same function, the same description."""
    server = MCPServer(SERVER_NAME, instructions="Tools for the LoDA document archive.", log_level="WARNING")
    for tool in tools:
        server.add_tool(tool.function, name=tool.name, description=tool.description)
    return server


def serve(server: MCPServer):
    """Over stdio by default (a client starts us), or over HTTP with --http."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="serve over HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8000)
    flags = parser.parse_args()
    if flags.http:
        server.run(transport="streamable-http", port=flags.port)
    else:
        server.run()


if __name__ == "__main__":
    serve(build_server(TOOLS))
