"""Unit tests for loda.adapters.authorization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from loda.adapters.authorization import DEPARTMENT_TOOLS, DEVELOPER_TOOLS, allowed_tools, is_developer
from loda.core.auth import User


def _user(groups):
    return User("u1", "Test User", "u@corp.example", tuple(groups))


def test_is_developer_true_for_admin_group():
    import archive
    user = _user([archive.ADMIN_GROUP])
    assert is_developer(user)


def test_is_developer_false_for_department_user():
    user = _user(["Finance-Records"])
    assert not is_developer(user)


def test_allowed_tools_developer_gets_extra_tool():
    import archive
    user = _user([archive.ADMIN_GROUP])
    tools = allowed_tools(user)
    assert "list_workspace_access" in tools
    assert DEPARTMENT_TOOLS.issubset(tools)


def test_allowed_tools_department_user_cannot_list_access():
    user = _user(["Finance-Records"])
    tools = allowed_tools(user)
    assert "list_workspace_access" not in tools
    assert tools == DEPARTMENT_TOOLS


def test_developer_tools_is_superset_of_department_tools():
    assert DEPARTMENT_TOOLS.issubset(DEVELOPER_TOOLS)
    assert len(DEVELOPER_TOOLS) == len(DEPARTMENT_TOOLS) + 1
