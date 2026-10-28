"""
Pattern 01b, hardened · the same pipeline, with the engineering a real split needs

Same agents, tools and code stage as labs/01b_sequential_agents.py (imported,
not copied). What's added is what makes a multi-agent system safe to run:

    1. Cost per agent      calls, tokens and seconds per agent, printed at the end
    2. Typed handoffs      each agent's final answer is checked against a schema
                           and invariants in code; a bad one goes back to the agent
    3. Checkpoints         the work order is saved after every stage, a crashed
                           run resumes where it stopped, and actions are idempotent
    4. Grounding check     every work item must quote the request it came from;
                           an item that can't is held for the user to confirm

Design notes: docs/01b_fixes_in_our_example.md

Run it:
    uv run labs/01b_sequential_agents_hardened.py
    uv run labs/01b_sequential_agents_hardened.py --demo-broken-handoff
    uv run labs/01b_sequential_agents_hardened.py --stop-after retrieval
    uv run labs/01b_sequential_agents_hardened.py --resume <order_id>
    uv run labs/01b_sequential_agents_hardened.py --resume <order_id> --from action
"""

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import shared

base = importlib.import_module("01b_sequential_agents")  # the simple version

RUNS = Path(__file__).parent.parent / "runs"
STAGES = ["intake", "retrieval", "policy", "action", "responder"]


# ── 2. Typed handoffs: the contract between each pair of stages ──────────────

class Contract(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)  # 2025 and "2025" are both fine


class Item(Contract):
    id: str
    action: Literal["retrieve", "delete"]
    doc_id: str = ""
    vendor: str = ""
    doc_type: str = ""
    year: str = ""
    quarter: str = ""
    description: str
    source: Literal["explicit", "implied"]
    evidence: str


class IntakeOut(Contract):
    items: list[Item] = Field(min_length=1)


class Resolved(Contract):
    item_id: str
    status: Literal["found", "ambiguous", "not_found"]
    doc_ids: list[str]
    note: str = ""


class RetrievalOut(Contract):
    resolved: list[Resolved]


class Decision(Contract):
    doc_id: str
    action: Literal["retrieve", "delete"]
    decision: Literal["allowed", "blocked", "needs_approval"]
    reason: str
    citation: str = ""
    notify: list[str] = []
    security_flag: bool = False


class PolicyOut(Contract):
    decisions: list[Decision]


def check_intake(final) -> dict:
    out = IntakeOut.model_validate(final)
    ids = [item.id for item in out.items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"item ids must be unique, got {ids}")
    return out.model_dump()


def check_retrieval(order: dict):
    """Schema, plus invariants only code can check: real ids, every item answered, right type."""
    def check(final) -> dict:
        out = RetrievalOut.model_validate(final)
        items = {item["id"]: item for item in order["items"]}
        problems = []
        answered = {r.item_id for r in out.resolved}
        problems += [f"item {i} was not resolved" for i in items if i not in answered]
        for r in out.resolved:
            if r.item_id not in items:
                problems.append(f"item {r.item_id} does not exist")
                continue
            wanted = items[r.item_id]["doc_type"]
            for doc_id in r.doc_ids:
                doc = base.find_document(doc_id)
                if doc is None:
                    problems.append(f"doc {doc_id} (item {r.item_id}) is not in the archive")
                elif wanted and doc["type"] != wanted:
                    problems.append(f"doc {doc_id} is a {doc['type']}, item {r.item_id} asked for {wanted}")
        if problems:
            raise ValueError("\n".join(problems))
        return out.model_dump()
    return check


def check_policy(order: dict):
    """Schema, plus: every request gets exactly one decision, and nothing else does."""
    def check(final) -> dict:
        out = PolicyOut.model_validate(final)
        asked = {(r["doc_id"], r["action"]) for r in order["requests"]}
        got = [(d.doc_id, d.action) for d in out.decisions]
        problems = [f"no decision for {doc_id} {action}" for doc_id, action in sorted(asked - set(got))]
        problems += [f"decision for {doc_id} {action}, which was not requested"
                     for doc_id, action in sorted(set(got) - asked)]
        if problems:
            raise ValueError("\n".join(problems))
        return out.model_dump()
    return check


# ── 4. Grounding check: every item must come from the request ────────────────

INTAKE_EVIDENCE = (
    '\nFor every item also give "evidence": a short quote, copied word for '
    "word, from the user's message or the attachment, that asks for it."
)


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,'\"").lower()


def grounding_check(order: dict) -> None:
    sources = _normal(order["message"] + " " + base.read_attachment("auditor_letter.txt"))
    for item in order["items"]:
        quote = _normal(item["evidence"])
        item["grounded"] = bool(quote) and quote in sources
        if not item["grounded"]:
            shared.step("Grounding", f"item {item['id']} ({item['description']}) quotes "
                                     f"{item['evidence']!r}, which is not in the request: held for the user")
    shared.step("Grounding", f"{sum(i['grounded'] for i in order['items'])}/{len(order['items'])} items traced to the request")


def hold_ungrounded(order: dict) -> None:
    """An item nobody asked for can't be acted on without the user saying yes (stricter wins)."""
    ungrounded = {i["id"] for i in order["items"] if not i["grounded"]}
    item_of = {(r["doc_id"], r["action"]): r["item_id"] for r in order["requests"]}
    for d in order["decisions"]:
        if item_of.get((d["doc_id"], d["action"])) in ungrounded and d["decision"] == "allowed":
            d["decision"] = "needs_approval"
            d["reason"] = "not traceable to the request; confirm it is wanted"
            d["notify"] = [*d["notify"], base.USER]


# ── 3. Checkpoints and idempotent actions ────────────────────────────────────

def save(run_dir: Path, order: dict, stage: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{STAGES.index(stage) + 1}_{stage}.json").write_text(json.dumps(order, indent=2), encoding="utf-8")
    shared.step("Checkpoint", f"saved after {stage}")


def load(run_dir: Path, before: str | None) -> tuple[dict, str]:
    """The work order from the last checkpoint (or the one just before `before`), and the next stage."""
    saved = sorted(run_dir.glob("*_*.json"), key=lambda p: int(p.name.split("_")[0]))
    if before:
        saved = [p for p in saved if int(p.name.split("_")[0]) <= STAGES.index(before)]
    if not saved:
        sys.exit(f"No checkpoint to resume from in {run_dir}")
    last = saved[-1].stem.split("_", 1)[1]
    next_stage = STAGES[STAGES.index(last) + 1] if last != STAGES[-1] else None
    shared.step("Checkpoint", f"resuming after {last}")
    return json.loads(saved[-1].read_text(encoding="utf-8")), next_stage


def idempotent(run_dir: Path, order_id: str):
    """Wrap 01b's carry_out: each (order, doc, action) happens once, however often the stage runs."""
    log_file = run_dir / "actions.json"
    done = json.loads(log_file.read_text(encoding="utf-8")) if log_file.exists() else {}

    def act(doc, action, final, notify, exports):
        key = f"{order_id}:{doc['doc_id']}:{action}"
        if key in done:
            if done[key] == "added to export package":
                exports.append(doc["doc_id"])  # already in the package; keep it listed
            return f"{done[key]} (already done in an earlier attempt, not repeated)"
        done[key] = base.carry_out(doc, action, final, notify, exports)
        log_file.write_text(json.dumps(done, indent=2), encoding="utf-8")
        return done[key]
    return act


# ── The pipeline: same order as 01b, each stage checked and checkpointed ─────

def run_stage(stage: str, order: dict, run_dir: Path) -> None:
    if stage == "intake":
        base.intake_agent(order, INTAKE_EVIDENCE, validate=check_intake)
        grounding_check(order)
    elif stage == "retrieval":
        base.retrieval_agent(order, validate=check_retrieval(order))
    elif stage == "policy":
        base.policy_agent(order, validate=check_policy(order))
    elif stage == "action":
        hold_ungrounded(order)
        base.access_and_action(order, act=idempotent(run_dir, order["order_id"]))
    elif stage == "responder":
        order["reply"] = base.responder_agent(order)


def demo_broken_handoff() -> None:
    """Show the Retrieval -> Policy contract rejecting a bad answer, without calling a model."""
    order = {"items": [{"id": "A", "doc_type": "invoice"}, {"id": "B", "doc_type": ""}]}
    broken = {"resolved": [{"item_id": "A", "status": "found", "doc_ids": ["1240", "9999"]}]}
    print("A Retrieval answer as a model might send it:\n ", json.dumps(broken))
    try:
        check_retrieval(order)(broken)
    except ValueError as error:
        print("\nRejected at the handoff, before Policy ever sees it:")
        print("\n".join(f"  - {line}" for line in str(error).splitlines()))
    malformed = {"resolved": [{"item_id": "A", "status": "found", "doc_ids": "1234"}]}
    print("\nAnd a malformed one:\n ", json.dumps(malformed))
    try:
        check_retrieval(order)(malformed)
    except ValueError as error:
        print("\nRejected:")
        print("\n".join(f"  {line}" for line in str(error).splitlines() if "errors.pydantic.dev" not in line))
    print("\nIn a real run the error goes back to the agent, which fixes its answer.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--demo-broken-handoff", action="store_true",
                        help="show a bad handoff being rejected (no model calls) and exit")
    parser.add_argument("--stop-after", choices=STAGES[:-1], help="simulate a crash after this stage")
    parser.add_argument("--resume", metavar="ORDER_ID", help="continue a saved run")
    parser.add_argument("--from", dest="from_stage", choices=STAGES[1:],
                        help="with --resume: re-run from this stage")
    args = parser.parse_args()

    if args.demo_broken_handoff:
        demo_broken_handoff()
        return

    if args.resume:
        run_dir = RUNS / args.resume
        order, next_stage = load(run_dir, args.from_stage)
        if next_stage is None:
            print("This run already finished. Use --from <stage> to re-run part of it.")
            return
    else:
        order_id = time.strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS / order_id
        order, next_stage = {"order_id": order_id, "user": base.USER, "message": base.MESSAGE}, STAGES[0]
        shared.step("Checkpoint", f"order {order_id}, saving to {run_dir.relative_to(RUNS.parent)}")

    for stage in STAGES[STAGES.index(next_stage):]:
        run_stage(stage, order, run_dir)
        save(run_dir, order, stage)
        if stage == args.stop_after:
            print(f"\nStopped after {stage} (simulated crash). Continue with:\n"
                  f"  --resume {order['order_id']}")
            shared.print_usage()
            return

    print(f"\nPriya sees:\n{order['reply']}")
    shared.print_usage()


if __name__ == "__main__":
    main()
