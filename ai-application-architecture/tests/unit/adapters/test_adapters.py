"""Unit tests for the concrete adapter implementations in loda.adapters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import pytest

from loda.adapters.entra_auth import ALREADY_EXPIRED, IdentityProvider
from loda.adapters.sqlite_store import ConversationStore, IN_MEMORY_STORE
from loda.core.auth import Identity, NotSignedIn
from loda.core.store import ConversationStore as ConversationStoreProtocol


# ---------- sqlite_store ----------

def test_sqlite_store_satisfies_protocol():
    store = ConversationStore(IN_MEMORY_STORE)
    assert isinstance(store, ConversationStoreProtocol)


def test_store_loads_empty_for_new_session():
    store = ConversationStore(IN_MEMORY_STORE)
    assert store.load("priya") == []


def test_store_saves_and_loads_messages():
    store = ConversationStore(IN_MEMORY_STORE)
    history = [{"role": "user", "content": "hello"}]
    store.save("priya", history)
    assert store.load("priya") == history


def test_store_save_is_idempotent_for_already_stored_messages():
    store = ConversationStore(IN_MEMORY_STORE)
    history = [{"role": "user", "content": "hello"}]
    store.save("priya", history)
    store.save("priya", history)  # second save must not duplicate
    assert len(store.load("priya")) == 1


def test_store_appends_new_messages_only():
    store = ConversationStore(IN_MEMORY_STORE)
    first = [{"role": "user", "content": "hello"}]
    store.save("priya", first)
    full = first + [{"role": "assistant", "content": "hi"}]
    store.save("priya", full)
    assert store.load("priya") == full


def test_store_forget_clears_session():
    store = ConversationStore(IN_MEMORY_STORE)
    store.save("priya", [{"role": "user", "content": "hello"}])
    store.forget("priya")
    assert store.load("priya") == []


def test_store_sessions_are_independent():
    store = ConversationStore(IN_MEMORY_STORE)
    store.save("priya", [{"role": "user", "content": "from priya"}])
    store.save("arun", [{"role": "user", "content": "from arun"}])
    assert store.load("priya")[0]["content"] == "from priya"
    assert store.load("arun")[0]["content"] == "from arun"


# ---------- entra_auth ----------

def test_identity_provider_satisfies_protocol():
    assert isinstance(IdentityProvider(), Identity)


def test_sign_in_returns_login_with_correct_user():
    provider = IdentityProvider()
    login = provider.sign_in("priya")
    assert login.user.user_id == "priya"
    assert login.token != ""


def test_verify_returns_user_from_valid_token():
    provider = IdentityProvider()
    login = provider.sign_in("priya")
    user = provider.verify(login.token)
    assert user.user_id == "priya"


def test_verify_raises_not_signed_in_for_expired_token():
    provider = IdentityProvider()
    login = provider.sign_in("priya", valid_for=ALREADY_EXPIRED)
    with pytest.raises(NotSignedIn):
        provider.verify(login.token)


def test_verify_raises_not_signed_in_for_tampered_token():
    provider = IdentityProvider()
    login = provider.sign_in("priya")
    with pytest.raises(NotSignedIn):
        provider.verify(login.token + "x")


def test_sign_in_unknown_user_raises_not_signed_in():
    provider = IdentityProvider()
    with pytest.raises(NotSignedIn):
        provider.sign_in("nobody-known")
