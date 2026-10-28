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
