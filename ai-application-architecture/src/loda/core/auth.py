from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class NotSignedIn(Exception):
    """No valid token — the question must not reach the model."""


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    email: str
    groups: tuple[str, ...]


@dataclass(frozen=True)
class Login:
    user: User
    token: str


@runtime_checkable
class Identity(Protocol):
    def sign_in(self, user_id: str) -> Login: ...
    def verify(self, token: str) -> User: ...
