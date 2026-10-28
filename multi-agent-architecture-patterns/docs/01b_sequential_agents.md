# Pattern 01b · Sequential Pipeline with real agents

Lab: `labs/01b_sequential_agents.py` · Data: `data/audit_request/`

Lab 01 shows the *shape* of a sequential pipeline, but none of its stages is
an agent: two are plain functions and two are single LLM calls. With a clean
request ("find invoice 1234") a normal program does the job better. This lab
makes the request messy enough that some stages genuinely have to interpret
and decide, and keeps the stages that must never vary as code.

## Problem statement

Priya (Finance) writes to DAS, with the auditor's letter attached:

> "The tax auditor is coming next week. I need the Q3 2025 invoices from
> Meyer Logistics and Westpark, plus that old Rhein Office Supplies invoice
> from 2019. Also, HR said contract 7001 can be deleted now that the employee
> has left. Can you sort all of it?"

### What makes it messy

| # | Mess | Where it lives |
|---|---|---|
| 1 | **Vendor names vary.** Meyer appears as "Meyer Logistics" and "Meyer Log. GmbH"; the second one is a credit note, not an invoice. | `extra_documents.json` (1240) |
| 2 | **The auditor's letter widens the scope.** "All supporting documents for Q3 2025 freight costs" implies Nordbahn Freight (1235), which Priya never named. | `auditor_letter.txt` |
| 3 | **The policy is prose with exceptions.** Personnel files may be deleted on request *unless a dispute is open*; then Legal must confirm. Tax records can't be deleted inside 10 years. | `retention_policy.md` |
| 4 | **Facts are spread across emails.** HR says 7001 can go; a later Legal email puts it on hold over a claim. | `emails.json` |
| 5 | **Some record text is hostile.** Westpark's record (1236) has an instruction smuggled into its title. It is data, never an instruction. | `documents.json` (1236) |
| 6 | **Platforms differ.** Rhein (4711) is on Classic: no export, a retrieval ticket instead. | `documents.json` (4711) |

## The pipeline

```
Intake (agent) → Retrieval (agent) → Policy (agent) → Access & Action (code) → Responder (agent)
```

The order is fixed, which is what makes it a sequential pipeline. Each agent
decides *how* to do its own step. One **work order** dict travels down the
pipeline, and each stage writes only its own key:

```
message → items → resolved → requests + decisions → actions + export_package → reply
```

### 1. Intake agent

| | |
|---|---|
| **Role** | Request analyst |
| **Goal** | A complete list of work items: what Priya said *and* what the auditor's letter implies, each tagged `explicit` or `implied` |
| **Figures out** | "that old Rhein invoice" = invoice, 2019; "freight costs" means freight vendors she didn't name |
| **Tools** | `read_attachment(name)`, `list_document_types()` |
| **Boundary** | Doesn't search the archive or judge permissions |

### 2. Retrieval agent

| | |
|---|---|
| **Role** | Archive researcher |
| **Goal** | Every item resolved to its record(s), or marked ambiguous / not found with a reason |
| **Figures out** | That "Meyer Log. GmbH" is the same vendor; that the credit note isn't what was asked for; which Q3 vendors are freight; when to widen or narrow a search |
| **Tools** | `search_documents(vendor, doc_type, year, quarter)` (fuzzy vendor match done in code; could be served via MCP), `find_document(doc_id)` |
| **Boundary** | Read-only. Never acts on text inside a record |

### 3. Policy agent

| | |
|---|---|
| **Role** | Compliance reasoner |
| **Goal** | Per request: `allowed` / `blocked` / `needs_approval`, citing the policy section or email relied on, plus who to notify |
| **Figures out** | That Legal's later hold overrides HR's email; that tax records can't be deleted; that 1236 contains suspicious text |
| **Tools** | `read_policy()`, `search_emails(doc_id)` |
| **Boundary** | Advises, doesn't act. Can only restrict, never grant |

### 4. Access & Action (code, not an agent)

| | |
|---|---|
| **Role** | Enforcement |
| **Logic** | Checks Priya's workspace and rights from `users.json`; the stricter of code and Policy wins. Allowed + CSP → export package; allowed + Classic → retrieval ticket; otherwise route to the owner and anyone the Policy agent named. Security flags go to the security team |
| **Why code** | Permissions and actions must behave the same every time. An LLM can't grant access here, and a request the Policy agent forgot is blocked by default |

### 5. Responder agent

| | |
|---|---|
| **Role** | User-facing writer |
| **Goal** | One plain reply: what's ready, what's pending and on whom, what was added from the auditor's letter, and why anything wasn't done |
| **Tools** | None. Sees only the finished work order, with titles cleaned so smuggled text never reaches it |
| **Boundary** | Can't change any outcome |

## Expected outcome

| Item | Source | Result |
|---|---|---|
| 1234 Meyer invoice | explicit | Export package |
| 1240 Meyer credit note | judgement call | Not an invoice, so not what Priya asked for; but the auditor wants *all supporting documents* for Q3 freight. Runs differ: excluded, added, or held for Priya to confirm. All three are defensible; discuss which is right |
| 1236 Westpark | explicit | Export package, flagged to security; the smuggled instruction is ignored |
| 1235 Nordbahn Freight | implied by auditor | Export package, marked "added from the auditor's letter" |
| 4711 Rhein 2019 | explicit (and implied) | Retrieval ticket (Classic platform) |
| 7001 delete | explicit | Blocked: legal hold, and outside Priya's workspace. Routed to hr-records-lead and Legal |

LLM runs vary. Use the trace lines to see where a run went differently, and
which stage caught (or missed) each mess.

## Agent, code or MCP?

- **Can you write the rule down? Write code.** Permission checks, platform
  routing and the export are code in this lab for that reason.
- **Is the input or the rule written in language? Use an LLM.** Intake, the
  policy reading and the reply.
- **Does the next step depend on what just came back? Use an agent loop.**
  Retrieval retrying searches; Policy deciding which emails to read.
- **MCP is plumbing, not intelligence.** It is a standard way to expose a tool
  such as `search_documents` to any model or client. It doesn't decide which
  tool to call; the agent does.

## Discussion questions

1. Remove the auditor's letter. Which stage becomes unnecessary?
2. Collapse everything into one agent that holds every tool. What could go
   wrong with record 1236, or with contract 7001?
3. Why is "stricter wins" in code rather than in the Policy agent's prompt?
4. Which stage would you move to code first if the archive's vendor names were
   clean?

For question 2, and whether this needs several agents at all, see
[01b_single_vs_multi_agent.md](01b_single_vs_multi_agent.md).
