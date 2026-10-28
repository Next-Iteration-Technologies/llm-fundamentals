# Lab 01b · The fixes, in our own example

Companion to [01b_making_multi_agent_work.md](01b_making_multi_agent_work.md).
That note lists the general fixes; this one shows what each looks like in the
auditor-request pipeline. Four are **built** in
`labs/01b_sequential_agents_hardened.py`, which imports the simple 01b and
adds only the hardening. The rest are sketches.

## 1. Tokens and latency

**Built: cost per agent.** Both 01b and the hardened version print a table at
the end. A live run on Claude Opus 5.5:

```
Agent              calls  tokens in  tokens out  seconds
Intake Agent           2      1,209       1,437       14
Retrieval Agent        8     18,440       1,595       28
Policy Agent           8     20,796       2,463       37
Responder Agent        1      1,609         848       10
Total                 19     42,054       6,343       89
```

Retrieval and Policy are 90% of the input tokens. Both re-send their growing
conversation on every tool call. That's where the fixes below would pay off.

Sketches:

- **Compact handoffs.** Policy currently receives each full record, including
  1236's title with the injected instruction. It only needs `doc_id`,
  `action`, `workspace`, `retention_class` and the title as stored: fewer
  tokens and less hostile text reaching the agent.
- **Caching.** `retention_policy.md` and each agent's system prompt are the
  same on every run, so prompt caching makes them about 10× cheaper from the
  second run on.
- **A model per stage.** Intake and Responder are simple language tasks and
  could run on Haiku; Policy, which must work out that Legal's hold overrides
  HR, stays on Opus.
- **Fan-out inside Retrieval.** Its 7 searches run one after another. One small
  Retrieval call per item, in parallel, would be roughly 6× faster.
- **A batch tool.** `search_documents(queries=[...])` would take all searches
  in one call: one step instead of seven, and less context re-sent.

## 2. State and handoffs

**Built: typed handoffs, checked in code.** Each agent's final answer is
validated against a schema (pydantic models `IntakeOut`, `RetrievalOut`,
`PolicyOut`) plus invariants only code can check:

- Retrieval: every item is answered, every `doc_id` exists in the archive, and
  a record's type matches the type the item asked for.
- Policy: every request gets exactly one decision, and nothing that wasn't
  requested does.

A failing answer doesn't crash the run or slip through. The error goes back to
the agent, which fixes it:

```
[Retrieval Agent] handoff rejected: item C was not resolved (+1 more)
[Retrieval Agent] done (handoff validated)
[Policy Agent] handoff rejected: no decision for 1236 retrieve
[Policy Agent] done (handoff validated)
```

In the simple 01b, a skipped decision is blocked by default; here it is sent
back to be decided.

Sketches:

- **Provenance.** `requested_as` is the first step. The full version keeps
  Priya's original words and the auditor's sentence on every item, so the
  Responder can never say "the work order doesn't mention an auditor's
  letter" (which happened in an early run).

## 3. Compounding errors

**Built: grounding check after Intake.** Intake must give, for each item, a
short quote from Priya's message or the auditor's letter that asks for it.
Code checks the quote really is there. An item that can't be traced is not
dropped; it is held for Priya to confirm (stricter wins):

```
[Grounding] item C (Westpark invoice) quotes 'please also send the Westpark file', which is not in the request: held for the user
[Grounding] 2/3 items traced to the request
...
[Access & Action] 1236 retrieve: not done; routed to finance-records-lead, priya
```

What this does and doesn't catch: it stops Intake inventing work nobody asked
for. Credit note 1240 is a different kind of error, a Retrieval judgement:
"all supporting documents" genuinely asks for it. The type-mismatch invariant
in section 2 catches it only when the item names a type, such as "invoice".

Sketches:

- **Later stages can object.** Give Policy the original request too; seeing
  "credit note, but Priya asked for invoices", it could return
  `needs_confirmation` instead of approving silently.
- **Bounded send-back.** Policy returns an `ambiguous` item to Retrieval once
  with a hint ("only Q3, only invoices"); if still unclear, it asks Priya.
- **Per-stage evals.** Five fixed inputs for Retrieval alone (e.g. "Meyer Log.
  GmbH" → 1234; "Q3 freight" → 1234 and 1235), run after every prompt change.

## 4. Distributed-system problems

**Built: checkpoints, resume and idempotent actions.**

- After every stage the work order is saved to
  `runs/<order_id>/<n>_<stage>.json` (`runs/` is git-ignored).
- `--stop-after <stage>` simulates a crash. `--resume <order_id>` continues
  from the last checkpoint without re-running finished agents. In the live run
  above, Intake and Retrieval ran once, the process stopped, and the resume ran
  only Policy, Action and Responder.
- Every side effect is logged in `runs/<order_id>/actions.json` under a key
  built from order id, document and action. Re-running the action stage
  repeats nothing:

```
[Access & Action] 1234 retrieve: added to export package (already done in an earlier attempt, not repeated)
```

Sketches:

- **Timeouts, retries with backoff, a dead-letter queue**, and a durable
  workflow engine (Temporal, Step Functions, LangGraph checkpoints) instead of
  JSON files.
- **Versioning.** Version prompts and schemas together, with contract tests
  between neighbouring stages.
- **Replay tests.** Save a real run's model replies and replay them after code
  changes, at no API cost.

## Try it

```
uv run labs/01b_sequential_agents_hardened.py                          # full run, cost table at the end
uv run labs/01b_sequential_agents_hardened.py --demo-broken-handoff    # contract rejecting bad answers, no model calls
uv run labs/01b_sequential_agents_hardened.py --stop-after retrieval   # simulated crash
uv run labs/01b_sequential_agents_hardened.py --resume <order_id>      # carry on from the checkpoint
uv run labs/01b_sequential_agents_hardened.py --resume <order_id> --from action   # idempotency
```

Run `uv run labs/01b_sequential_agents.py` too, and compare the cost tables:
the hardening adds checks and files, not model calls, unless a handoff is
rejected.
