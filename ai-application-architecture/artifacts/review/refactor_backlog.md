# Refactor Backlog — SOLID + Modularization

Generated: 2026-10-26 | Reviewer: review-mode (NI) | Stack: Python

---

## Critical

### RB-001 · DIP violation — high-level modules depend on concrete step files

**File:** `step14_capstone.py:20–25`, `step13_logging_traces.py:30–38`
**Severity:** Critical
**What:** `step14` and `step13` import directly from `step1`, `step4`, `step6`, `step8`, `step9`, `step12` — concrete modules, not abstractions.
**Why:** Violates DIP (core-standards: "Depend on abstractions, not concretions"). Any change to a low-level step forces changes in every higher step.
**Impact if ignored:** Cascade of edits across 9 files for any interface change. Impossible to swap implementations (e.g. replace SQLite store with Redis).
**Fix:** Introduce `Protocol` ABCs (`LLMModel`, `ConversationStore`, `Tracer`) in `src/loda/core/`. All steps depend on those, not each other.

---

### RB-002 · Parallel Inheritance Chain (Shotgun Surgery)

**File:** `step10:99` → `step11:95` → `step12:109` → `step13:121`
**Severity:** Critical
**What:** Each step's `build()` wraps the previous step's `build()`. Adding any new capability requires editing every step above it.
**Why:** Shotgun Surgery smell (core-standards). Violates OCP — existing code must be modified to add a feature.
**Fix:** Replace manual wrapping chain with a `Pipeline.compose([features])` that applies capabilities in order via a single composition point.

---

### RB-003 · Duplicated `answer()` / `build()` signatures

**File:** `step8:130,135`, `step9:52`, `step10:99,104`, `step11:95`, `step12:109,114`, `step13:121,127`
**Severity:** Critical
**What:** `build(llm, login, connection) -> Chatbot` and `answer(bot, question) -> str` are re-declared in 6 files with identical signatures but different wrapping behavior.
**Why:** Duplicated code smell + DRY violation (core-standards).
**Fix:** One `answer()` in `pipeline.py`; per-step logic moves to `Feature` protocol implementations.

---

## High

### RB-004 · SRP violation — `step8_authentication.py` has 5 responsibilities

**File:** `step8_authentication.py` (170 lines)
**Severity:** High
**What:** Single file handles: JWT token creation, token verification, user/group modelling, tool parameter binding, `Chatbot` dataclass, `build()` wiring, `answer()` execution, CLI demo.
**Why:** SRP (core-standards: "Each class does one thing only").
**Fix:** Split into `auth.py` (identity), `chatbot.py` (dataclass), `agent.py` (loop wiring).

---

### RB-005 · Message chain — `bot.model.tracer.tools_called`

**File:** `step13_logging_traces.py:155`
**Severity:** High
**What:** `bot.model.tracer.tools_called` — three-level chain to reach a property.
**Why:** Message Chains smell (core-standards: "a.getB().getC().getD()").
**Fix:** Expose `bot.tools_called` property on `Chatbot`; `Tracer` stays internal to `TracingModel`.

---

### RB-006 · `Chatbot` dataclass is a Data Class smell

**File:** `step8_authentication.py:116–128`
**Severity:** High
**What:** `Chatbot` holds `model`, `tools`, `store`, `login`, `system` — no behaviour, just data. All behaviour lives in free functions that reach into it.
**Why:** Data Class smell (core-standards). Low cohesion.
**Fix:** Move `Chatbot` to `src/loda/chatbot.py`; add `answer(question)` method so behaviour lives with data.

---

### RB-007 · `step9_mcp_server_secure.py` mixes auth + tool implementations

**File:** `step9_mcp_server_secure.py` (117 lines)
**Severity:** High
**What:** Authorization logic (`allowed_tools`, `can_open`, `is_developer`) lives alongside tool implementations (`search_document`, `get_retention_info`, etc.).
**Why:** SRP violation — two reasons to change: auth rules change OR tool signatures change.
**Fix:** Split into `authorization.py` (rules) and `tools_secure.py` (tool implementations).

---

## Medium

### RB-008 · Magic string `":memory:"` repeated

**File:** `step13_logging_traces.py:153`, `step14_capstone.py:46`
**Severity:** Medium
**What:** `":memory:"` used directly as a SQLite path in two places.
**Why:** Magic literal smell (core-standards: "Replace magic numbers/literals with named constants").
**Fix:** Constant `IN_MEMORY_STORE = ":memory:"` in `sqlite_store.py`.

---

### RB-009 · Long method — `run_golden_questions()`

**File:** `step13_logging_traces.py:146–160`
**Severity:** Medium
**What:** Single method does: load cases, loop, build bot, run answer, check case, print result, aggregate score.
**Why:** Long Method smell + SRP (core-standards: "Small, focused methods").
**Fix:** Extract `run_case(case, llm, connection)` and `print_summary(passed, total)`.

---

### RB-010 · Implicit step ordering — educational scaffold leaking into production chain

**File:** All `stepN_*.py` files
**Severity:** Medium
**What:** The numbered file names encode execution order as a naming convention. `step14` must always import `step13`, never `step12` directly.
**Why:** Implicit dependency / Logical Dependency smell (core-standards: "avoid logical dependency").
**Fix:** New `src/loda/` package has no numbered files. Steps kept as isolated demos importing from `loda.*`.

---

## Low

### RB-011 · `__getattr__` magic delegation on `TracingModel`

**File:** `step13_logging_traces.py:102–106`
**Severity:** Low
**What:** `TracingModel.__getattr__` delegates unknown attributes to wrapped model. Silent at definition time; fails at runtime.
**Why:** Implicit coupling; hard to discover `redactor` is accessible from outside.
**Fix:** Explicit `@property def redactor` on `TracingModel`.

---

### RB-012 · `print()` used for runtime output in production code

**File:** `step13_logging_traces.py:132`, `step14_capstone.py:43,51`
**Severity:** Low
**What:** `print()` used for trace notifications and capstone progress.
**Why:** Python guideline: "Use `logging` (not `print()`) for any runtime output".
**Fix:** Replace with `logger = logging.getLogger(__name__)`.

---

*Total issues: 12 (3 Critical · 4 High · 3 Medium · 2 Low)*
