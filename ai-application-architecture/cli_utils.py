"""Shared utilities for all demo scripts."""

import argparse
import itertools
import sys
from collections.abc import Callable, Iterator

import llm_client
from request_view import show_request

from loda.adapters.entra_auth import ALREADY_EXPIRED, TOKEN_LIFETIME, IdentityProvider
from loda.adapters.mcp_tools import McpConnection
from loda.core.auth import NotSignedIn

EXIT_WORDS = {"exit", "quit", "bye"}
_call_numbers = itertools.count(1)


def read_questions(llm: llm_client.Session) -> Iterator[str]:
    while True:
        question = llm.read_question()
        if question is None or question.lower() in EXIT_WORDS:
            print("Goodbye!")
            return
        if question:
            yield question


def show_call(messages, reply, system=None, tools=None):
    show_request(next(_call_numbers), messages, reply, system=system, tools=tools)


def run_safely(chat: Callable[[], None]):
    """Run a chat; a setup problem ends with advice instead of a stack trace."""
    try:
        chat()
    except llm_client.SetupProblem as problem:
        sys.exit(f"\n{problem}")


def run_signed_in_chat(
    name: str,
    demo_lines: list[str],
    server,
    build_fn: Callable,
    answer_fn: Callable,
    identity: IdentityProvider | None = None,
) -> None:
    """Sign in, connect to the MCP server, build the chatbot, run the chat loop."""
    identity = identity or IdentityProvider()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--user", default="priya")
    parser.add_argument("--expired", action="store_true")
    flags, _ = parser.parse_known_args()
    llm = llm_client.Session(name, demo_lines)
    try:
        login = identity.sign_in(flags.user, ALREADY_EXPIRED if flags.expired else TOKEN_LIFETIME)
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
