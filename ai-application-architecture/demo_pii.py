"""Demo: PII redaction — names, emails and phone numbers replaced before reaching the model."""

import llm_client
from demo_utils import run_safely, run_signed_in_chat
from loda.adapters.authorization import is_developer
from loda.adapters.entra_auth import IdentityProvider
from loda.adapters.mcp_tools import DEFAULT_SERVER, McpConnection, tools_from_server
from loda.adapters.sqlite_store import ConversationStore
from loda.core.auth import Login
from loda.features.gateway import ModelGateway, private_model
from loda.features.guardrails import BLOCKED_INPUT_ANSWER, SCOPE_RULES, check_output, looks_like_injection, wrap_as_outside_data
from loda.features.pii import Redactor, RedactingModel, personal_data_visible_to
from loda.features.retrieval import SYSTEM_PROMPT
from loda.pipeline.agent import answer_with_memory
from loda.pipeline.builder import acting_as, offered_tools, with_token
from loda.pipeline.chatbot import Chatbot

DEMO_LINES = [
    "Hi, I'm Priya Sharma. Where is invoice 1234?",
    "Who can see the Finance workspace?",
]

_identity = IdentityProvider()


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    system = SYSTEM_PROMPT + f" The signed-in user belongs to: {', '.join(login.user.groups)}." + SCOPE_RULES
    tools = [wrap_as_outside_data(t) for t in
             with_token(offered_tools(acting_as(login.user, tools_from_server(connection)), login.user), login)]
    redactor = Redactor()
    model = RedactingModel(ModelGateway(default=llm, private=private_model(llm)), redactor)
    return Chatbot(llm, login, tools, system, ConversationStore(), model=model)


def answer(bot: Chatbot, question: str) -> str:
    if looks_like_injection(question):
        return BLOCKED_INPUT_ANSWER
    _identity.verify(bot.login.token)
    reply = answer_with_memory(bot.llm, bot.store, bot.session_id, question,
                               tools=bot.tools, system=bot.system, model=bot.model)
    return check_output(bot.model.redactor.restore(reply, personal_data_visible_to(bot.login.user, is_developer)))


if __name__ == "__main__":
    run_safely(lambda: run_signed_in_chat("demo_pii", DEMO_LINES, DEFAULT_SERVER, build, answer, _identity))
