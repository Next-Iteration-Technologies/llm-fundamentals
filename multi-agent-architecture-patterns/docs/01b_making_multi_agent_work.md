# Lab 01b · Making multi-agent work

Companion to [01b_single_vs_multi_agent.md](01b_single_vs_multi_agent.md),
which lists the costs of splitting into agents. This note is the other half:
once you have a real reason to split (scale, ownership, audit), how do you
handle those costs?

You shouldn't fear multi-agent. Its problems are the same ones microservices
have, and they have known fixes. The point is: **don't split for free; when
you do split, budget for the engineering.**

## 1. Tokens and latency

- **Pass compact, structured handoffs, not transcripts.** Each agent gets only
  the fields it needs, not the previous agent's reasoning.
- **Cache what doesn't change.** System prompts and the policy document are
  prime candidates for prompt caching. They're the same on every run.
- **Pick the model per stage.** A cheap, fast model for Intake; a strong one
  for Policy.
- **Parallelise inside a stage.** Resolve each item concurrently rather than in
  one long loop (Pattern 03, fan-out, inside Pattern 01).
- **Design tools to need fewer steps.** For example, `search_documents` could
  take a list of queries, so it's one call instead of seven.
- **Clear old tool results** from the context once a step has used them.

## 2. State and handoffs

- **Typed contracts at every boundary.** Define a schema for each stage's
  output, have the model return structured output, and validate it in code.
  Invalid output gets repaired or rejected, never passed on silently.
- **One work order, with each stage writing only its own fields.** The lab
  already does this.
- **Carry provenance.** The original request, and *why* each item exists,
  travel with it. `requested_as` is a small version of this.
- **Write each role's boundaries into its prompt.** "Decide X, not Y" — the
  Classic-platform fix.
- **Check invariants in code.** Every request gets a decision; every decision
  maps to a real record. "Skipped means blocked" is one of these.

## 3. Compounding errors

- **Put a check between stages:** a cheap code check, plus a verifier agent
  where it's worth it.
- **Let later stages object.** Give them the original request and allow them
  to return "this doesn't match what was asked", instead of trusting earlier
  output blindly.
- **Use confidence and escalation.** Low-confidence items go to a human or
  back to the user (Pattern 04).
- **Allow bounded send-backs:** Policy can return an item to Retrieval at most
  once.
- **Evaluate each stage separately.** Measure every stage's accuracy and fix
  the weakest one, rather than staring at end-to-end failures.

## 4. Distributed-system problems

- **Run it on a durable workflow engine** (Temporal, Step Functions, LangGraph
  checkpoints, or similar). Save the work order after every stage and resume
  from the last checkpoint after a failure.
- **Make side effects idempotent.** Give each export or ticket an idempotency
  key, for example built from the work order id and doc id, so a retry doesn't
  do it twice. Keep all side effects in the final code stage, as the lab does.
- **Set timeouts and retries with backoff, and use a dead-letter queue** for
  work orders that keep failing.
- **Version prompts and schemas together,** with contract tests between
  neighbouring stages.
- **Trace everything.** Give each work order one trace id, and log every
  stage's input, output, tokens and time. `step()` is the minimal version.
- **Test in three layers:** stage tests with fixed inputs (like a scripted dry
  run), replays of recorded real runs, and end-to-end evals.

## Where the lab stands

| Fix | Simple 01b | Hardened 01b |
|---|---|---|
| Each stage writes only its own fields | Yes | Yes |
| Side effects only in the code stage | Yes | Yes |
| Invariant checks in code | Partly ("skipped means blocked", stricter wins) | Yes, at every handoff |
| Cost per agent (calls, tokens, time) | Yes | Yes |
| Schema validation with send-back | No | Yes |
| Grounding check on Intake | No | Yes |
| Checkpoints, resume, idempotent actions | No | Yes |
| Trace | Minimal (`step()`) | Minimal (`step()`) |
| Per-stage evals, caching, model per stage, fan-out | No | No |

What each of these looks like in our pipeline, and how to run the hardened
version: [01b_fixes_in_our_example.md](01b_fixes_in_our_example.md).

## For the session

> "Multi-agent is a distributed system with an LLM inside each node. Treat it
> like one — contracts, checkpoints, idempotency, tracing, evals — and the
> costs become manageable."
