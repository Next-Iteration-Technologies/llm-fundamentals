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
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import jwt

import archive
import llm_client
from step1_bare_model_call import read_questions, run_safely
from step2_agent_loop import Tool
from step3_retrieval import SYSTEM_PROMPT as STEP3_PROMPT
from step4_conversation_store import ConversationStore, answer_with_memory
from step6_mcp_client import STEP6_SERVER, McpConnection, tools_from_server

TRAINING_SECRET = "loda-training-only-never-use-in-production"   # Entra ID signs with its own keys
AUDIENCE = "loda-chatbot"
TOKEN_LIFETIME = timedelta(hours=1)
ALREADY_EXPIRED = timedelta(seconds=-1)
IDENTITY_PARAMETERS = ("requested_by", "uploaded_by", "raised_by")   # set from the token, hidden from the model
DEMO_LINES = [
    "Hi, I'm Priya. Did my upload from yesterday get archived?",
    "Please request access to employment contract 7001 for me. I need it for an audit.",
]


class NotSignedIn(Exception):
    """No valid token: the question must not reach the model."""


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


class IdentityProvider:
    """Stands in for Entra ID: signs a token at sign-in, and checks it on every request."""

    def __init__(self, secret: str = TRAINING_SECRET):
        self._secret = secret

    def sign_in(self, user_id: str, valid_for: timedelta = TOKEN_LIFETIME) -> Login:
        record = archive.find_user(user_id)
        if record is None:
            raise NotSignedIn(f"There is no user called {user_id}.")
        user = User(record["user_id"], record["name"], record["email"], tuple(record["groups"]))
        now = datetime.now(timezone.utc)
        claims = {"sub": user.user_id, "name": user.name, "email": user.email, "groups": list(user.groups),
                  "aud": AUDIENCE, "iat": now, "exp": now + valid_for}
        return Login(user, jwt.encode(claims, self._secret, algorithm="HS256"))

    def verify(self, token: str) -> User:
        try:
            claims = jwt.decode(token, self._secret, algorithms=["HS256"], audience=AUDIENCE)
        except jwt.ExpiredSignatureError as error:
            raise NotSignedIn("Your sign-in has expired. Please sign in again.") from error
        except jwt.InvalidTokenError as error:
            raise NotSignedIn("This sign-in is not valid.") from error
        return User(claims["sub"], claims["name"], claims["email"], tuple(claims["groups"]))


IDENTITY_PROVIDER = IdentityProvider()


# ---------- the identity goes to the tools from the token, not from the model ----------

def bind_parameters(tool: Tool, fixed: dict) -> Tool:
    """The same tool, with some arguments set by our code. The model no longer sees them."""
    fixed = {name: value for name, value in fixed.items() if name in tool.parameters.get("properties", {})}
    if not fixed:
        return tool
    visible = {
        **tool.parameters,
        "properties": {name: schema for name, schema in tool.parameters["properties"].items() if name not in fixed},
        "required": [name for name in tool.parameters.get("required", []) if name not in fixed],
    }
    return replace(tool, parameters=visible, function=lambda **arguments: tool.function(**{**arguments, **fixed}))


def acting_as(user: User, tools: list[Tool]) -> list[Tool]:
    return [bind_parameters(tool, {name: user.user_id for name in IDENTITY_PARAMETERS}) for tool in tools]


# ---------- the signed-in chatbot: later steps replace one part at a time ----------

@dataclass(frozen=True)
class Chatbot:
    """Everything one signed-in chat needs."""
    llm: llm_client.Session        # questions, replay, running tools
    login: Login
    tools: list[Tool]
    system: str
    store: ConversationStore
    model: object = None           # what .chat() goes to; None means the llm itself

    @property
    def session_id(self) -> str:
        return f"{self.login.user.user_id}-chat"


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    system = STEP3_PROMPT + f" The signed-in user belongs to: {', '.join(login.user.groups)}."
    return Chatbot(llm, login, acting_as(login.user, tools_from_server(connection)), system, ConversationStore())


def answer(bot: Chatbot, question: str) -> str:
    IDENTITY_PROVIDER.verify(bot.login.token)       # every question: an expired sign-in stops here
    return answer_with_memory(bot.llm, bot.store, bot.session_id, question, tools=bot.tools, system=bot.system, model=bot.model)


def login_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--user", default="priya")
    parser.add_argument("--expired", action="store_true")
    flags, _ = parser.parse_known_args()
    return flags


def run_signed_in_chat(step_name: str, demo_lines: list[str], server,
                       build_chatbot: Callable[..., Chatbot], answer_question: Callable[[Chatbot, str], str]):
    """Sign in, connect to the MCP server, build the chatbot, chat. Steps 9 to 13 reuse this."""
    flags = login_flags()
    llm = llm_client.Session(step_name, demo_lines)
    try:
        login = IDENTITY_PROVIDER.sign_in(flags.user, ALREADY_EXPIRED if flags.expired else TOKEN_LIFETIME)
    except NotSignedIn as problem:
        sys.exit(f"Sign-in failed: {problem}")
    with McpConnection(server) as connection:
        bot = build_chatbot(llm, login, connection)
        print(f"Signed in as {login.user.name} ({', '.join(login.user.groups)}) on {llm.provider_name}. "
              f"Type 'exit' to stop.\n")
        for question in read_questions(llm):
            try:
                print(f"LLM: {answer_question(bot, question)}\n")
            except NotSignedIn as problem:
                print(f"Refused before the model was called: {problem}\n")
                return


if __name__ == "__main__":
    run_safely(lambda: run_signed_in_chat("step8_authentication", DEMO_LINES, STEP6_SERVER, build, answer))
