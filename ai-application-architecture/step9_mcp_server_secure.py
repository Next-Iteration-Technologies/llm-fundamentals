"""
Step 9, the server: the LoDA MCP server, now checking who calls.

Every tool takes the caller's token as one more argument. On every call the
server checks the token (who is this?) and then what this user may do: which
tools, and which workspaces. This is the check that counts, because every
client goes through it: the chatbot, n8n, Copilot, or a script that skips the
chatbot entirely. The model never decides who may see what.

The rules:
  - Department users (Finance-Records, HR-Records) use the six chatbot tools and
    see documents in their own workspace only.
  - Developers (LoDA-Developers) see every workspace and may also list who has access.
  - A refusal reveals nothing: no title, no folder, only that access can be requested.

Start it like the step 6 server: the chatbot starts it over stdio, or --http.
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import archive
import step3_retrieval
from step3_retrieval import TOOLS
from step6_mcp_server import SERVER_NAME, serve
from step8_authentication import IDENTITY_PROVIDER, NotSignedIn, User

DEPARTMENT_TOOLS = frozenset(tool.name for tool in TOOLS)
DEVELOPER_TOOLS = DEPARTMENT_TOOLS | {"list_workspace_access"}
NOT_VISIBLE = {
    "found": "not visible to you",
    "message": "This document is in a workspace you cannot open. You can raise an access request; "
               "the workspace owner decides.",
}


def is_developer(user: User) -> bool:
    return archive.ADMIN_GROUP in user.groups


def allowed_tools(user: User) -> frozenset[str]:
    return DEVELOPER_TOOLS if is_developer(user) else DEPARTMENT_TOOLS


def can_open(user: User, workspace: str) -> bool:
    return is_developer(user) or archive.WORKSPACE_GROUP.get(workspace) in user.groups


def caller(token: str, tool_name: str) -> User:
    """Authentication, then authorization: who is calling, and may they use this tool?"""
    try:
        user = IDENTITY_PROVIDER.verify(token)
    except NotSignedIn as problem:
        raise ToolError(str(problem)) from problem
    if tool_name not in allowed_tools(user):
        raise ToolError(f"{user.name} may not use {tool_name}.")
    return user


def description_of(tool_name: str) -> str:
    return next(tool.description for tool in TOOLS if tool.name == tool_name)


server = MCPServer(f"{SERVER_NAME}-secure", instructions="Tools for the LoDA document archive.", log_level="WARNING")


@server.tool(description=description_of("search_document"))
def search_document(doc_id: str, caller_token: str) -> dict:
    user = caller(caller_token, "search_document")
    result = archive.search_document(doc_id)
    return result if not result["found"] or can_open(user, result["workspace"]) else NOT_VISIBLE


@server.tool(description=description_of("get_retention_info"))
def get_retention_info(doc_id: str, caller_token: str) -> dict:
    user = caller(caller_token, "get_retention_info")
    document = archive.find_document(doc_id)
    if document is not None and not can_open(user, document["workspace"]):
        return NOT_VISIBLE
    return archive.get_retention_info(doc_id)


@server.tool(description=description_of("raise_access_request"))
def raise_access_request(doc_id: str, reason: str, caller_token: str) -> dict:
    user = caller(caller_token, "raise_access_request")      # anyone may ask; the owner decides
    return archive.raise_access_request(doc_id, reason, requested_by=user.user_id)


@server.tool(description=description_of("check_upload_status"))
def check_upload_status(caller_token: str, upload_id: str = "") -> dict:
    user = caller(caller_token, "check_upload_status")
    result = archive.check_upload_status(uploaded_by=user.user_id, upload_id=upload_id)
    visible = [upload for upload in result["uploads"] if can_open(user, upload["workspace"])]
    return {"uploads": visible} if visible else {"uploads": [], "note": "No upload found."}


@server.tool(description=description_of("search_known_issues"))
def search_known_issues(question: str, caller_token: str) -> dict:
    caller(caller_token, "search_known_issues")
    return step3_retrieval.search_known_issues(question)


@server.tool(description=description_of("handoff_to_developers"))
def handoff_to_developers(summary: str, caller_token: str, urgency: str = "normal") -> dict:
    user = caller(caller_token, "handoff_to_developers")
    return archive.handoff_to_developers(summary, urgency, raised_by=user.user_id)


@server.tool(description="List everyone who can open a workspace, and through which group. "
                         "Contains names and email addresses. Developers only.")
def list_workspace_access(workspace: str, caller_token: str) -> dict:
    caller(caller_token, "list_workspace_access")
    return archive.list_workspace_access(workspace)


if __name__ == "__main__":
    serve(server)
