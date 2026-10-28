# Lab 01b · Would a single agent do?

Companion to [01b_sequential_agents.md](01b_sequential_agents.md).

## Short answer

Yes, almost certainly. A capable model with every tool and the whole problem
in one prompt would very likely reach the same five outcomes. There are only
seven records, a one-page policy and three emails, so nothing here is too big
for one context window.

## Most of the safety is code, not the split

- "Stricter decision wins", the workspace and rights check, and "skipped
  means blocked" all live in plain Python (`access_and_action`).
- No agent has a delete tool, so contract 7001 can't be deleted however
  confused a model gets.
- The instruction smuggled into record 1236 can't send anything, because no
  agent has a send tool.

So **one agent plus the same code gate** gets about 90% of what this lab
shows.

## What the split actually buys you

| Benefit | How much it matters here |
|---|---|
| **Independent check.** The Policy agent didn't do the searching, so it isn't motivated to approve what Retrieval found. A single agent judges its own work. | Real, but small at this size |
| **Least privilege.** Retrieval can't read emails; Policy can't search. Injected text in a record only reaches agents without action tools. | Real, but the code gate already covers the dangerous actions |
| **Context isolation.** Each agent sees only its slice. | Not needed: everything fits in one context |
| **Testable stages.** You can check Retrieval's output on its own. | Useful for teaching and debugging |
| **A model per stage.** A cheap model for Intake, a strong one for Policy. | Not used in the lab |
| **Visible trace.** Learners can see who decided what. | The main teaching value |

## When one agent stops being enough

- **Scale:** thousands of records, a 200-page policy, whole mailboxes. One
  context fills up and quality drops.
- **Ownership:** Compliance owns and audits the Policy agent's prompt; another
  team owns retrieval.
- **Regulation:** an auditor wants proof that the approver and the fetcher
  were separate. The split is then a requirement, not a design choice.
- **Untrusted content at volume:** the more hostile text an agent reads, the
  more you want it unable to act.

## Costs of the split

Multi-agent isn't free. This lab already ran into most of these costs.

### Tokens and latency

- **Context gets re-sent every step.** `run_agent` sends the whole
  conversation so far on each step. Retrieval made 7 calls, so the earliest
  tool results were billed 7 times.
- **Information gets re-read.** Policy reads the policy and the emails
  itself, and the Responder sees a summary of everything again. A single
  agent would read each once.
- **Overhead per agent.** Each one has its own system prompt and tool
  protocol.
- **It's slow.** A full run is about 20 sequential LLM calls; a single agent
  might need about 12. Each stage waits for the previous one, so nothing runs
  in parallel.

### State and handoffs

Every handoff is a **contract**: a schema plus an understanding of who owns
what. Three of the bugs found while building this lab were at handoffs:

- **The Responder said "the work order doesn't mention an auditor's letter".**
  The information existed but wasn't passed on, so `requested_as` was added.
- **The Policy agent treated "Classic" as a policy issue** because its
  boundary was unclear. A prompt change fixed it.
- **In a dry run, a request with no decision fell through the gap** between
  stages, so "skipped means blocked" was added.

A single agent has no handoffs, so none of these could have happened.

### Errors compound

- **Mistakes carry forward.** Intake's extra "supporting documents" item is
  why credit note 1240 appeared. Later stages trust earlier output and don't
  re-read the original request, so they rarely catch it.
- **Five stages at 95% each give about 77% end to end.**

### Distributed-system problems

This lab runs in one process, so it avoids most of these. A real deployment
wouldn't:

- **Partial failure:** if Policy times out, Intake's and Retrieval's work is
  lost unless the work order is saved at each step.
- **Retries and duplicates:** retrying the action stage could export twice or
  open two tickets.
- **Versioning:** changing one agent's prompt or output format can break the
  next agent without any error.
- **Debugging:** you need tracing across agents to see where a run went
  wrong. The `step()` trace lines are the lab's minimal version.
- **Testing:** you need tests per stage *and* end to end.

## The trade-off

| | Single agent + code gate | Multi-agent + code gate |
|---|---|---|
| Tokens / latency | Lower | Higher |
| Handoff bugs | None | Every boundary |
| Failure handling | One loop | Checkpoints and retries per stage |
| Independent check | No | Yes |
| Scale / ownership / audit separation | Weak | Strong |

**Rule of thumb:** start with one agent plus code. Split only when a specific
need forces it: context limits, separate ownership, an auditor requiring
separation, or untrusted content. Don't split just because a diagram looks
cleaner.

When you do have a reason to split, see
[01b_making_multi_agent_work.md](01b_making_multi_agent_work.md) for how to
handle each of these costs.

## For the session

Learners will spot this, so say it up front:

> "This would work with one agent. We split it because each split is a
> boundary you'll need at scale or under audit, and the code gate is what
> actually keeps it safe."

## Optional experiment

Build a single-agent version (`01c`) with the same tools and the same code
gate, and run it side by side with 01b. Learners would see the same outcomes,
with fewer calls and no independent check. That shows the trade-off better
than a slide does. (Not built yet.)
