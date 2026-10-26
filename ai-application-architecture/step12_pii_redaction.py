"""
Step 12: PII redaction.

Routing personal data to the local model (step 11) helps. Safer still: the
model never sees it. Before every call, names, email addresses and phone
numbers are swapped for placeholders like [PERSON_1]. The same person always
gets the same placeholder, so the model can still reason ("[PERSON_1] and
[PERSON_2] see Finance through Finance-Records"). The mapping stays in our
code for this chat. Names are put back only for users allowed to see them:
developers see everyone, a department user sees only their own name.

With nothing personal left in the text, the gateway can use the company model again.

Try it:
    uv run step12_pii_redaction.py --user mia --demo      # names restored for a developer
    uv run step12_pii_redaction.py --user priya --demo    # Priya's own name only

    You: Hi, I'm Priya Sharma. Where is invoice 1234?
    You: Who can see the Finance workspace?
"""

import archive
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
from loda.features.pii import Redactor, RedactingModel, personal_data_visible_to
from loda.features.retrieval import SYSTEM_PROMPT as STEP3_PROMPT
from loda.pipeline.agent import answer_with_memory
from loda.pipeline.builder import acting_as, offered_tools, with_token
from loda.pipeline.chatbot import Chatbot

DEMO_LINES = [
    "Hi, I'm Priya Sharma. Where is invoice 1234?",
    "Who can see the Finance workspace?",
]


def build(llm: llm_client.Session, login: Login, connection: McpConnection) -> Chatbot:
    system = STEP3_PROMPT + f" The signed-in user belongs to: {', '.join(login.user.groups)}." + SCOPE_RULES
    tools = [wrap_as_outside_data(t) for t in
             with_token(offered_tools(acting_as(login.user, tools_from_server(connection)), login.user), login)]
    redactor = Redactor()
    model = RedactingModel(ModelGateway(default=llm, private=private_model(llm)), redactor)
    return Chatbot(llm, login, tools, system, ConversationStore(), model=model)


def answer(bot: Chatbot, question: str) -> str:
    if looks_like_injection(question):
        return BLOCKED_INPUT_ANSWER
    IDENTITY_PROVIDER.verify(bot.login.token)
    reply = answer_with_memory(bot.llm, bot.store, bot.session_id, question,
                               tools=bot.tools, system=bot.system, model=bot.model)
    is_dev = lambda u: archive.ADMIN_GROUP in u.groups
    restored = bot.model.redactor.restore(reply, personal_data_visible_to(bot.login.user, is_dev))
    return check_output(restored)


if __name__ == "__main__":
    run_safely(lambda: run_signed_in_chat("step12_pii_redaction", DEMO_LINES, SECURE_SERVER, build, answer))
