from typing import Protocol, runtime_checkable

from loda.core.auth import Login


@runtime_checkable
class Tracer(Protocol):
    @property
    def tools_called(self) -> list[str]: ...

    def start(self, login: Login, question: str) -> None: ...
    def write(self, kind: str, **fields: object) -> None: ...
