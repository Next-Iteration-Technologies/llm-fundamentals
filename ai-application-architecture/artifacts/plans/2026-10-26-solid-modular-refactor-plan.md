# SOLID + Modular Refactor — Implementation Plan

**Date:** 2026-10-26
**Branch:** solid-modular-refactor
**Backlog:** `artifacts/review/refactor_backlog.md`
**Stack:** Python 3.12 · pytest

---

## Target Structure

```
src/
  loda/
    core/           ← Protocols / ABCs only (no concrete code)
      model.py      LLMModel protocol
      store.py      ConversationStore protocol
      tools.py      Tool protocol + ToolCall dataclass
      auth.py       User, Login, Identity protocols
      tracer.py     Tracer protocol
    adapters/       ← One concrete impl per protocol
      anthropic_llm.py   wraps llm_client.Session
      sqlite_store.py    ConversationStore on SQLite
      mcp_tools.py       Tool adapter over McpConnection
      entra_auth.py      JWT-based IdentityProvider
    features/       ← One file per capability (OCP)
      retrieval.py       known-issues search
      guardrails.py      injection + output checks
      pii.py             Redactor + RedactingModel
      gateway.py         ModelGateway routing
      tracing.py         Tracer + TracingModel
    pipeline/
      builder.py    Pipeline.compose([features]) → Chatbot
      chatbot.py    Chatbot dataclass + answer() method
tests/
  unit/             mirrors src/loda/ structure
  integration/      full-stack tests (existing test_full_chatbot.py moves here)
steps/              educational demos — import from loda.*
```

---

## Phases

### Phase 1 — Core Protocols (DIP foundation)
**Fixes:** RB-001, RB-006
**Files created:**
- `src/loda/core/model.py` — `LLMModel` Protocol (`chat(messages, system, tools) -> Reply`)
- `src/loda/core/store.py` — `ConversationStore` Protocol (`load`, `save`, `forget`)
- `src/loda/core/tools.py` — `Tool` dataclass + `ToolCall` Protocol
- `src/loda/core/auth.py` — `User`, `Login`, `Identity` Protocol
- `src/loda/core/tracer.py` — `Tracer` Protocol (`start`, `write`, `tools_called`)
- `src/loda/pipeline/chatbot.py` — `Chatbot` dataclass with `answer(question) -> str`

**Tests:** `tests/unit/core/test_protocols.py` — isinstance checks against mock impls

---

### Phase 2 — Adapters (concrete implementations)
**Fixes:** RB-001, RB-004
**Files created:**
- `src/loda/adapters/anthropic_llm.py` — wraps `llm_client.Session`, implements `LLMModel`
- `src/loda/adapters/sqlite_store.py` — moves `ConversationStore` from `step4`; adds `IN_MEMORY_STORE` constant (fixes RB-008)
- `src/loda/adapters/mcp_tools.py` — moves `McpConnection` + `tools_from_server` from `step6`
- `src/loda/adapters/entra_auth.py` — moves `IdentityProvider`, `User`, `Login` from `step8`

**Tests:** `tests/unit/adapters/` — unit tests per adapter using `scripted_llm.py`

---

### Phase 3 — Features (one capability per file)
**Fixes:** RB-002, RB-003, RB-007, RB-011, RB-012
**Files created:**
- `src/loda/features/retrieval.py` — `search_known_issues` + `SYSTEM_PROMPT`
- `src/loda/features/guardrails.py` — `looks_like_injection`, `check_output`, `as_outside_data`
- `src/loda/features/pii.py` — `Redactor`, `RedactingModel`, `personal_data_visible_to`
- `src/loda/features/gateway.py` — `ModelGateway`, `contains_personal_data`, `private_model`
- `src/loda/features/tracing.py` — `Tracer`, `TracingModel`, `traced`; explicit `@property redactor` (fixes RB-011); `logging` replaces `print` (fixes RB-012)

**Tests:** `tests/unit/features/` — unit tests per feature; existing tests in `tests/` updated to import from `loda.*`

---

### Phase 4 — Pipeline (composition replaces wrapping chain)
**Fixes:** RB-002, RB-003, RB-005, RB-009
**Files created:**
- `src/loda/pipeline/builder.py` — `Pipeline` with `compose(features) -> Chatbot`; `bot.tools_called` property (fixes RB-005)
- `src/loda/pipeline/chatbot.py` updated — `answer(question)` method; no free `answer()` functions elsewhere

**Extract:** `run_case()` and `print_summary()` from `run_golden_questions` (fixes RB-009)

**Tests:** `tests/integration/test_full_chatbot.py` — existing 7-request capstone script

---

### Phase 5 — Step file migration
**Fixes:** RB-010
**Change:** Each `stepN_*.py` imports from `loda.*` instead of `stepM_*.py`.
Steps become thin demos; no production logic left in them.

**No new tests** — existing tests must still pass after import rewire.

---

### Phase 6 — Authorization split
**Fixes:** RB-007
**Files:**
- `src/loda/adapters/authorization.py` — `allowed_tools`, `can_open`, `is_developer`
- `step9_mcp_server_secure.py` shrinks to tool implementations only, imports from `loda.adapters.authorization`

**Tests:** `tests/unit/adapters/test_authorization.py`

---

## Definition of Done (per NI code-generation skill)

- All phases implemented sequentially
- `pytest tests/` passes 100% after each phase
- No `print()` in `src/loda/` (use `logging`)
- No cross-step imports in `src/loda/` (`from stepN_*` anywhere is a failure)
- `mypy src/loda/` passes with no errors

---

## Not in scope

- Changing `llm_client.py`, `archive.py`, `request_view.py` — treated as external libs
- Rewriting `step6_mcp_server.py` — MCP server wire protocol unchanged
- UI changes (`web_app.py`, `web_static/`)
