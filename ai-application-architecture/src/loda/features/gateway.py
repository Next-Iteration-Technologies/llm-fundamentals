"""Model gateway feature: routes requests to the right model based on content."""

import logging

import llm_client
from loda.features.pii import PERSONAL_DATA

logger = logging.getLogger(__name__)

PRIVATE_PROVIDER = "ollama-chat"


def contains_personal_data(text: str) -> bool:
    return any(pattern.search(text) for pattern in PERSONAL_DATA.values())


def conversation_text(messages: list[dict], system: str | None) -> str:
    return "\n".join([system or "", *(message.get("content") or "" for message in messages)])


class ModelGateway:
    """One door, two models. Presents the same .chat() interface as a single model."""

    def __init__(self, default: llm_client.Session, private: llm_client.Session):
        self.default = default
        self.private = private

    def chat(self, messages: list[dict], system: str | None = None, tools: list[dict] | None = None):
        if contains_personal_data(conversation_text(messages, system)):
            return self._send(self.private, "personal data stays inside", messages, system, tools)
        try:
            return self._send(self.default, "no personal data", messages, system, tools)
        except llm_client.SetupProblem as problem:
            logger.warning("gateway: %s failed (%s)", self.default.provider_name, problem)
            return self._send(self.private, "fallback", messages, system, tools)

    def _send(self, route: llm_client.Session, reason: str, messages, system, tools):
        logger.info("gateway -> %s (%s)", route.provider_name, reason)
        return route.chat(messages, system=system, tools=tools)


def private_model(llm: llm_client.Session) -> llm_client.Session:
    return llm_client.Session(
        f"{llm.round_name}-private", [], provider_name=PRIVATE_PROVIDER
    )
