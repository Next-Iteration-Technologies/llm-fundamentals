"""LoDA MCP server: document archive tools with JWT-based caller authorization."""

import argparse

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import archive
from loda.adapters.authorization import allowed_tools, can_open
from loda.adapters.entra_auth import IdentityProvider
from loda.core.auth import NotSignedIn, User
from loda.features.retrieval import TOOLS
from loda.features.retrieval import search_known_issues as _search_known_issues

SERVER_NAME = "loda-archive"

_identity = IdentityProvider()

_NOT_VISIBLE = {
    "found": "not visible to you",
    "message": "This document is in a workspace you cannot open. "
               "You can raise an access request; the workspace owner decides.",
}


def _caller(token: str, tool_name: str) -> User:
    try:
        user = _identity.verify(token)
    except NotSignedIn as problem:
        raise ToolError(str(problem)) from problem
    if tool_name not in allowed_tools(user):
        raise ToolError(f"{user.name} may not use {tool_name}.")
    return user


def _description(tool_name: str) -> str:
    return next(tool.description for tool in TOOLS if tool.name == tool_name)


def build_server() -> MCPServer:
    """Construct the MCP server with all authorized archive tools."""
    server = MCPServer(SERVER_NAME, instructions="Tools for the LoDA document archive.", log_level="WARNING")

    @server.tool(description=_description("search_document"))
    def search_document(doc_id: str, caller_token: str) -> dict:
        user = _caller(caller_token, "search_document")
        result = archive.search_document(doc_id)
        return result if not result["found"] or can_open(user, result["workspace"]) else _NOT_VISIBLE

    @server.tool(description=_description("get_retention_info"))
    def get_retention_info(doc_id: str, caller_token: str) -> dict:
        user = _caller(caller_token, "get_retention_info")
        document = archive.find_document(doc_id)
        if document is not None and not can_open(user, document["workspace"]):
            return _NOT_VISIBLE
        return archive.get_retention_info(doc_id)

    @server.tool(description=_description("raise_access_request"))
    def raise_access_request(doc_id: str, reason: str, caller_token: str) -> dict:
        user = _caller(caller_token, "raise_access_request")
        return archive.raise_access_request(doc_id, reason, requested_by=user.user_id)

    @server.tool(description=_description("check_upload_status"))
    def check_upload_status(caller_token: str, upload_id: str = "") -> dict:
        user = _caller(caller_token, "check_upload_status")
        result = archive.check_upload_status(uploaded_by=user.user_id, upload_id=upload_id)
        visible = [upload for upload in result["uploads"] if can_open(user, upload["workspace"])]
        return {"uploads": visible} if visible else {"uploads": [], "note": "No upload found."}

    @server.tool(description=_description("search_known_issues"))
    def search_known_issues(question: str, caller_token: str) -> dict:
        _caller(caller_token, "search_known_issues")
        return _search_known_issues(question)

    @server.tool(description=_description("handoff_to_developers"))
    def handoff_to_developers(summary: str, caller_token: str, urgency: str = "normal") -> dict:
        user = _caller(caller_token, "handoff_to_developers")
        return archive.handoff_to_developers(summary, urgency, raised_by=user.user_id)

    @server.tool(
        description="List everyone who can open a workspace, and through which group. "
                    "Contains names and email addresses. Developers only."
    )
    def list_workspace_access(workspace: str, caller_token: str) -> dict:
        _caller(caller_token, "list_workspace_access")
        return archive.list_workspace_access(workspace)

    return server


def serve(server: MCPServer | None = None) -> None:
    """Serve over stdio (default) or HTTP with --http."""
    if server is None:
        server = build_server()
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
