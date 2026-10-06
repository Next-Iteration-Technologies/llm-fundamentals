import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import step9_mcp_server_secure as secure
from step8_authentication import IDENTITY_PROVIDER


def token_of(user_id):
    return IDENTITY_PROVIDER.sign_in(user_id).token


def test_a_department_user_sees_their_own_workspace_only():
    assert secure.search_document("1234", token_of("priya"))["found"] is True
    assert secure.search_document("7001", token_of("priya")) == secure.NOT_VISIBLE


def test_a_refusal_reveals_nothing_about_the_document():
    refusal = json.dumps(secure.search_document("1234", token_of("arun")))
    assert "Meyer" not in refusal and "Invoices" not in refusal


def test_only_developers_may_list_workspace_access():
    with pytest.raises(ToolError, match="may not use"):
        secure.list_workspace_access("Finance", token_of("priya"))
    assert secure.list_workspace_access("Finance", token_of("mia"))["people"]


def test_the_requester_comes_from_the_token():
    request = secure.raise_access_request("7001", "audit", token_of("priya"))
    assert request["requested_by"] == "priya"


def test_anyone_may_ask_for_access_but_only_the_owner_decides():
    request = secure.raise_access_request("1234", "audit", token_of("arun"))
    assert request["approver"] == "finance-records-lead"
