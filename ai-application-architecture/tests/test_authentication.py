import pytest

from step2_agent_loop import ARCHIVE_TOOLS
from step8_authentication import ALREADY_EXPIRED, IDENTITY_PROVIDER, IdentityProvider, NotSignedIn, acting_as


def test_a_token_says_who_signed_in():
    login = IDENTITY_PROVIDER.sign_in("priya")
    user = IDENTITY_PROVIDER.verify(login.token)
    assert user.user_id == "priya" and "Finance-Records" in user.groups


def test_an_expired_token_is_refused():
    login = IDENTITY_PROVIDER.sign_in("priya", ALREADY_EXPIRED)
    with pytest.raises(NotSignedIn, match="expired"):
        IDENTITY_PROVIDER.verify(login.token)


def test_a_token_signed_by_someone_else_is_refused():
    forged = IdentityProvider(secret="someone-else-entirely-different-secret").sign_in("mia")
    with pytest.raises(NotSignedIn):
        IDENTITY_PROVIDER.verify(forged.token)


def test_identity_comes_from_the_login_not_from_the_model():
    tools = {tool.name: tool for tool in acting_as(IDENTITY_PROVIDER.sign_in("arun").user, ARCHIVE_TOOLS)}
    access = tools["raise_access_request"]
    assert "requested_by" not in access.parameters["properties"]
    request = access.function(doc_id="1234", reason="audit", requested_by="priya")   # the model claims Priya
    assert request["requested_by"] == "arun"
