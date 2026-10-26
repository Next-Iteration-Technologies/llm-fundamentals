"""
Step 11: a model gateway.

Until now every message went to one model, whatever it contained. The gateway
is one door in front of two models:
  - the company model (the default provider, for example Nexus) for ordinary questions,
  - a local model (Ollama, inside our network) for anything with personal data.
If the company model fails, the gateway falls back to the local one. Personal
data never falls back the other way.

The chatbot does not notice: the gateway has the same .chat() as a model.
Watch the line "gateway -> ..." before every call.

Try it (needs Ollama running with the model in providers.toml [ollama-chat]):
    uv run step11_model_gateway.py --user mia --demo

    You: Where is invoice 1234?                      -> the company model
    You: Who can see the Finance workspace?          -> the local model (names and emails)
"""

from dataclasses import replace

import llm_client
from step1_bare_model_call import run_safely
from step8_authentication import IDENTITY_PROVIDER, run_signed_in_chat

from loda.adapters.mcp_tools import DEFAULT_SERVER as SECURE_SERVER, McpConnection, tools_from_server
from loda.adapters.sqlite_store import ConversationStore
from loda.core.auth import Login
from loda.features.gateway import ModelGateway, private_model
from loda.features.guardrails import (
    BLOCKED_INPUT_ANSWER, SCOPE_RULES, check_output, looks_like_injection, wrap_as_outside_data,
)
from loda.features.retrieval import SYSTEM_PROMPT as STEP3_PROMPT
from loda.pipeline.agent import answer_with_memory
from loda.pipeline.builder import acting_as, offered_tools, with_token
from loda.pipeline.chatbot import Chatbot

DEMO_LINES = [
    "Where is invoice 1234?",
    "Who can see the Finance workspace?",
]


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    system = STEP3_PROMPT + f" The signed-in user belongs to: {', '.join(login.user.groups)}." + SCOPE_RULES
    tools = [wrap_as_outside_data(t) for t in
             with_token(offered_tools(acting_as(login.user, tools_from_server(connection)), login.user), login)]
    model = ModelGateway(default=llm, private=private_model(llm))
    return Chatbot(llm, login, tools, system, ConversationStore(), model=model)


def answer(bot: Chatbot, question: str) -> str:
    if looks_like_injection(question):
        return BLOCKED_INPUT_ANSWER
    IDENTITY_PROVIDER.verify(bot.login.token)
    return check_output(answer_with_memory(bot.llm, bot.store, bot.session_id, question,
                                           tools=bot.tools, system=bot.system, model=bot.model))


if __name__ == "__main__":
    run_safely(lambda: run_signed_in_chat("step11_model_gateway", DEMO_LINES, SECURE_SERVER, build, answer))
