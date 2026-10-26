"""Demo MCP server: the LoDA archive tools over stdio (no auth, for local demos)."""

import argparse

from mcp.server.mcpserver import MCPServer

from loda.core.tools import Tool
from loda.features.retrieval import TOOLS

SERVER_NAME = "loda-archive"


def build_server(tools: list[Tool]) -> MCPServer:
    server = MCPServer(SERVER_NAME, instructions="Tools for the LoDA document archive.", log_level="WARNING")
    for tool in tools:
        server.add_tool(tool.function, name=tool.name, description=tool.description)
    return server


def serve(server: MCPServer | None = None) -> None:
    if server is None:
        server = build_server(TOOLS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="serve over HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8000)
    flags = parser.parse_args()
    if flags.http:
        server.run(transport="streamable-http", port=flags.port)
    else:
        server.run()


if __name__ == "__main__":
    serve()
