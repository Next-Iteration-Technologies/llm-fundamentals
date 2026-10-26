"""
Step 8: authentication.

"Hi, I'm Priya" proves nothing: anyone can type it. Until now the model filled
in requested_by and uploaded_by from whatever the user typed. Now the user
signs in first. Sign-in gives a signed token that says who they are and which
groups they are in. Our code checks the token on every question, and the
user's id goes into the tools from the token, never from the model.

IdentityProvider below stands in for Entra ID so the training runs offline.
In production the token comes from Entra ID and is checked against its
public keys; the rest of this file stays the same.

Try it:
    uv run step8_authentication.py --user priya --demo
    uv run step8_authentication.py --user arun --demo        # Arun types "I'm Priya": the tools still get arun
    uv run step8_authentication.py --user priya --expired    # an expired token never reaches the model
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from mcp import StdioServerParameters

import llm_client
from step1_bare_model_call import read_questions, run_safely

from loda.adapters.entra_auth import ALREADY_EXPIRED, IDENTITY_PARAMETERS, IdentityProvider, TOKEN_LIFETIME
from loda.adapters.mcp_tools import McpConnection, tools_from_server
from loda.adapters.sqlite_store import ConversationStore
from loda.core.auth import Login, NotSignedIn, User
from loda.core.tools import Tool
from loda.features.retrieval import SYSTEM_PROMPT as STEP3_PROMPT
from loda.pipeline.agent import answer_with_memory
from loda.pipeline.builder import acting_as, bind_parameters
from loda.pipeline.chatbot import Chatbot

DEMO_SERVER = StdioServerParameters(
    command=sys.executable, args=[str(Path(__file__).parent / "step6_mcp_server.py")]
)
DEMO_LINES = [
    "Hi, I'm Priya. Did my upload from yesterday get archived?",
    "Please request access to employment contract 7001 for me. I need it for an audit.",
]

IDENTITY_PROVIDER = IdentityProvider()


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    system = STEP3_PROMPT + f" The signed-in user belongs to: {', '.join(login.user.groups)}."
    return Chatbot(llm, login, acting_as(login.user, tools_from_server(connection)), system, ConversationStore())


def answer(bot: Chatbot, question: str) -> str:
    IDENTITY_PROVIDER.verify(bot.login.token)
    return answer_with_memory(bot.llm, bot.store, bot.session_id, question,
                              tools=bot.tools, system=bot.system, model=bot.model)


def login_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--user", default="priya")
    parser.add_argument("--expired", action="store_true")
    flags, _ = parser.parse_known_args()
    return flags


def run_signed_in_chat(step_name: str, demo_lines: list[str], server,
                       build_fn: Callable[..., Chatbot], answer_fn: Callable[[Chatbot, str], str]):
    """Sign in, connect to the MCP server, build the chatbot, chat. Steps 9–13 reuse this."""
    flags = login_flags()
    llm = llm_client.Session(step_name, demo_lines)
    try:
        login = IDENTITY_PROVIDER.sign_in(flags.user, ALREADY_EXPIRED if flags.expired else TOKEN_LIFETIME)
    except NotSignedIn as problem:
        sys.exit(f"Sign-in failed: {problem}")
    with McpConnection(server) as connection:
        bot = build_fn(llm, login, connection)
        print(f"Signed in as {login.user.name} ({', '.join(login.user.groups)}) on {llm.provider_name}. "
              f"Type 'exit' to stop.\n")
        for question in read_questions(llm):
            try:
                print(f"LLM: {answer_fn(bot, question)}\n")
            except NotSignedIn as problem:
                print(f"Refused before the model was called: {problem}\n")
                return


if __name__ == "__main__":
    run_safely(lambda: run_signed_in_chat("step8_authentication", DEMO_LINES, DEMO_SERVER, build, answer))
