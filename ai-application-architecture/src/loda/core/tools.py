from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    function: Callable[..., object]

    def definition(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def text_field(description: str) -> dict:
    return {"type": "string", "description": description}


def arguments_schema(required: dict, optional: dict | None = None) -> dict:
    return {
        "type": "object",
        "properties": {**required, **(optional or {})},
        "required": list(required),
    }
