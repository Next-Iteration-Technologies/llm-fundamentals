"""Pipeline builder: assembles a fully-featured Chatbot from composable parts."""

import logging
from dataclasses import replace

import archive

import llm_client
from loda.adapters.entra_auth import IDENTITY_PARAMETERS, IdentityProvider
from loda.adapters.mcp_tools import DEFAULT_SERVER, McpConnection, tools_from_server
from loda.adapters.sqlite_store import ConversationStore
from loda.core.auth import Login, User
from loda.core.tools import Tool
from loda.features.gateway import ModelGateway, private_model
from loda.features.guardrails import SCOPE_RULES, check_input, check_output, wrap_as_outside_data
from loda.features.pii import Redactor, RedactingModel, personal_data_visible_to
from loda.features.retrieval import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT
from loda.features.tracing import Tracer, TracingModel, traced
from loda.pipeline.agent import answer_with_memory
from loda.pipeline.chatbot import Chatbot

logger = logging.getLogger(__name__)

IDENTITY_PROVIDER = IdentityProvider()

# --- authorization helpers (Phase 6 will move to loda.adapters.authorization) ---

_DEPARTMENT_TOOLS = frozenset(
    tool_name
    for tool_name in (
        "search_document",
        "get_retention_info",
        "raise_access_request",
        "check_upload_status",
        "handoff_to_developers",
        "search_known_issues",
    )
)
_DEVELOPER_TOOLS = _DEPARTMENT_TOOLS | {"list_workspace_access"}


def _is_developer(user: User) -> bool:
    return archive.ADMIN_GROUP in user.groups


def _allowed_tools(user: User) -> frozenset[str]:
    return _DEVELOPER_TOOLS if _is_developer(user) else _DEPARTMENT_TOOLS


def bind_parameters(tool: Tool, fixed: dict) -> Tool:
    """Same tool with some arguments pre-filled by our code; the model no longer sees them."""
    fixed = {name: value for name, value in fixed.items() if name in tool.parameters.get("properties", {})}
    if not fixed:
        return tool
    visible = {
        **tool.parameters,
        "properties": {
            name: schema
            for name, schema in tool.parameters["properties"].items()
            if name not in fixed
        },
        "required": [
            name for name in tool.parameters.get("required", []) if name not in fixed
        ],
    }
    return replace(tool, parameters=visible, function=lambda **arguments: tool.function(**{**arguments, **fixed}))


def acting_as(user: User, tools: list[Tool]) -> list[Tool]:
    return [bind_parameters(tool, {name: user.user_id for name in IDENTITY_PARAMETERS}) for tool in tools]


def offered_tools(tools: list[Tool], user: User) -> list[Tool]:
    """Layer 1: only offer the model what this user may use. The server checks again anyway."""
    allowed = _allowed_tools(user)
    return [tool for tool in tools if tool.name in allowed]


def with_token(tools: list[Tool], login: Login) -> list[Tool]:
    """Every call carries the user's token, added by our code, invisible to the model."""
    return [bind_parameters(tool, {"caller_token": login.token}) for tool in tools]


# --- assembly ---

def build_chatbot(
    llm: llm_client.Session,
    login: Login,
    connection: McpConnection,
) -> Chatbot:
    """Build a fully composed Chatbot: auth, tools, guardrails, gateway, PII, tracing."""
    system = (
        BASE_SYSTEM_PROMPT
        + f" The signed-in user belongs to: {', '.join(login.user.groups)}."
        + SCOPE_RULES
    )

    raw_tools = tools_from_server(connection)
    authorized = with_token(offered_tools(acting_as(login.user, raw_tools), login.user), login)
    guarded_tools = [wrap_as_outside_data(tool) for tool in authorized]

    gateway = ModelGateway(default=llm, private=private_model(llm))
    redactor = Redactor()
    redacting = RedactingModel(gateway, redactor)
    tracer = Tracer(redactor)
    tracing = TracingModel(redacting, tracer)
    traced_tools = [traced(tool, tracer) for tool in guarded_tools]

    store = ConversationStore()

    return Chatbot(
        llm=llm,
        login=login,
        tools=traced_tools,
        system=system,
        store=store,
        model=tracing,
        pipeline_fn=_full_answer,
    )


# --- the single answer function (fixes RB-002, RB-003) ---

def _full_answer(bot: Chatbot, question: str) -> str:
    """Full pipeline: verify → guardrails → agent loop → PII restore → output check → trace."""
    IDENTITY_PROVIDER.verify(bot.login.token)

    blocked = check_input(question)
    if blocked:
        return blocked

    tracer = getattr(bot.model, "tracer", None)
    if tracer is not None:
        tracer.start(bot.login, question)

    reply = answer_with_memory(
        bot.llm,
        bot.store,
        bot.session_id,
        question,
        tools=bot.tools,
        system=bot.system,
        model=bot.model,
    )

    redactor = getattr(bot.model, "redactor", None)
    if redactor is not None:
        may_see = personal_data_visible_to(bot.login.user, _is_developer)
        reply = redactor.restore(reply, may_see)

    reply = check_output(reply)

    if tracer is not None:
        tracer.write("answer", answer=reply)
        logger.info("trace %s written to traces/%s", tracer.trace_id, tracer.file_name)

    return reply
