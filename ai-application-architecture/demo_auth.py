"""Demo: authentication — sign-in tokens; the user's id comes from the token, not what they type."""

import sys
from pathlib import Path

from mcp import StdioServerParameters

import llm_client
from demo_utils import run_safely, run_signed_in_chat
from loda.adapters.entra_auth import IdentityProvider
from loda.adapters.mcp_tools import McpConnection, tools_from_server
from loda.adapters.sqlite_store import ConversationStore
from loda.core.auth import Login
from loda.features.retrieval import SYSTEM_PROMPT
from loda.pipeline.agent import answer_with_memory
from loda.pipeline.builder import acting_as
from loda.pipeline.chatbot import Chatbot

DEMO_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / "demo_mcp_server.py")],
)
DEMO_LINES = [
    "Hi, I'm Priya. Did my upload from yesterday get archived?",
    "Please request access to employment contract 7001 for me. I need it for an audit.",
]

_identity = IdentityProvider()


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    system = SYSTEM_PROMPT + f" The signed-in user belongs to: {', '.join(login.user.groups)}."
    return Chatbot(llm, login, acting_as(login.user, tools_from_server(connection)), system, ConversationStore())


def answer(bot: Chatbot, question: str) -> str:
    _identity.verify(bot.login.token)
    return answer_with_memory(bot.llm, bot.store, bot.session_id, question,
                              tools=bot.tools, system=bot.system, model=bot.model)


if __name__ == "__main__":
    run_safely(lambda: run_signed_in_chat("demo_auth", DEMO_LINES, DEMO_SERVER, build, answer, _identity))
