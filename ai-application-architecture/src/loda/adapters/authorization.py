"""Authorization rules: which tools a user may call and which workspaces they may open."""

import archive
from loda.core.auth import User

DEPARTMENT_TOOLS = frozenset({
    "search_document",
    "get_retention_info",
    "raise_access_request",
    "check_upload_status",
    "handoff_to_developers",
    "search_known_issues",
})
DEVELOPER_TOOLS = DEPARTMENT_TOOLS | {"list_workspace_access"}


def is_developer(user: User) -> bool:
    return archive.ADMIN_GROUP in user.groups


def allowed_tools(user: User) -> frozenset[str]:
    return DEVELOPER_TOOLS if is_developer(user) else DEPARTMENT_TOOLS


def can_open(user: User, workspace: str) -> bool:
    return is_developer(user) or archive.WORKSPACE_GROUP.get(workspace) in user.groups
