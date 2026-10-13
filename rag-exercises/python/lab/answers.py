"""python rag.py answers RUN|--best: do better passages give better answers? (phase 6, the RAG triad)

For each question, the run's setup finds the chunks, the chat model (--chat) answers, the judge (--judge) scores:
  context relevance  did we send passages that fit?   from the gold passages: the run's Recall@5 (no model)
  faithfulness       is every claim backed by the passages?   the judge checks each claim
  answer relevance   does it answer the question?     the judge rates 1-5
plus two checks with no model: must say / must not say (Contoso's golden set) and refusals (questions with no answer).
Every reply is saved (lab/cache.py), so a rerun costs nothing.
"""
import re, sys, time

import models

from . import chunkers, datasets, llm, pipeline, results

REFUSED = re.compile(r"don.?t find|do not find|not (found|in the (passages|documents|sources|rulebook))|cannot (answer|find)|"
                     r"no (information|passage)|isn.?t (in|mentioned)|not mentioned|doesn.?t (say|mention)", re.I)
STOP = set("a an the of to in on for and or is are was be by with that this it its as at from we our".split())


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower()) if w not in STOP]


def says(answer, phrase, share=0.7):
    """A must-say phrase counts as said when most of its content words are in the answer (wording may differ)."""
    want = words(phrase)
    have = set(words(answer))
    return bool(want) and sum(w in have for w in want) / len(want) >= share


def spread(items, n):
    """n items spread evenly over the list, first and last included: a --limit sample that covers every level."""
    if not n or n >= len(items):
        return items
    if n == 1:
        return items[:1]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


FAITH = ("PASSAGES\n{passages}\n\nANSWER\n{answer}\n\nSplit the answer into its claims. For each claim, write one line: "
         "the claim, then SUPPORTED if the passages back it, or UNSUPPORTED if they don't. A statement that the passages "
         "don't contain the answer counts as SUPPORTED when that is true.")
RELEVANCE = ("QUESTION\n{question}\n\nANSWER\n{answer}\n\nDoes the answer actually answer the question? Rate 1 to 5: "
             "5 = answers all of it, 3 = answers part of it, 1 = doesn't answer it (an honest \"I don't find this\" to a "
             "question with no answer in the documents counts as 5). Reply with the number first, then one sentence why.")


def judge_faith(judge, passages, answer):
    r = llm.ask(judge, "You check answers against sources. Be strict.", FAITH.format(
        passages="\n".join(f"[{p['source']}] {p['text']}" for p in passages), answer=answer))
    s, u = len(re.findall(r"\bSUPPORTED\b", r["text"])), len(re.findall(r"\bUNSUPPORTED\b", r["text"]))
    return (round(s / (s + u), 4) if s + u else None), r


def judge_relevance(judge, question, answer):
    r = llm.ask(judge, "You rate answers.", RELEVANCE.format(question=question, answer=answer))
    m = re.search(r"\b([1-5])\b", r["text"])
    return (round((int(m.group(1)) - 1) / 4, 4) if m else None), r


def answer_one(ds, q, index, setup, chat, judge, system, gold_scores=None, passages=None):
    """Answer one question and score it. passages: given (final phase) instead of retrieved."""
    if passages is None:
        final, _ = pipeline.retrieve(index, q.text, setup, q=q)
        passages = [c for c, _ in final[:setup.k]]
    sent, held, where = pipeline.residency(passages, chat)
    prompt = pipeline.build_prompt(q.text, sent)
    r = llm.ask(chat, system, prompt)
    answer = r["text"]
    row = {"n": q.n, "id": q.id, "level": q.level, "answer": answer, "seconds": r["seconds"], "saved": bool(r["saved"]),
           "tokens_in": (r.get("usage") or {}).get("input_tokens") or chunkers.ntok(system) + chunkers.ntok(prompt),
           "held_back": len(held)}
    if gold_scores is not None:
        row["context"] = gold_scores.get("R@5")
    refused = bool(REFUSED.search(answer))
    row["refused"] = refused
    if q.refuse:
        row["refusal_ok"] = refused
    elif not q.gold and not q.reference:
        pass
    if q.must_say:
        row["must_say"] = round(sum(says(answer, p) for p in q.must_say) / len(q.must_say), 4)
    if q.must_not_say:
        row["must_not_say_ok"] = not any(says(answer, p, 1.0) for p in q.must_not_say)
    row["faithfulness"], fr = judge_faith(judge, sent, answer)
    row["answer_relevance"], rr = judge_relevance(judge, q.text, answer)
    row["judge_seconds"] = round(fr["seconds"] + rr["seconds"], 1)
    return row


def summarize(rows):
    def avg(k, vals=None):
        v = [r[k] for r in rows if r.get(k) is not None] if vals is None else vals
        v = [float(x) for x in v]
        return round(sum(v) / len(v), 4) if v else None
    return {"n": len(rows), "context": avg("context"), "faithfulness": avg("faithfulness"),
            "answer_relevance": avg("answer_relevance"), "must_say": avg("must_say"),
            "refusals_ok": avg("refusal_ok"), "refusal_questions": sum(1 for r in rows if "refusal_ok" in r),
            "wrong_refusals": sum(1 for r in rows if r.get("refused") and "refusal_ok" not in r),
            "tokens_in": avg("tokens_in"), "seconds": avg("seconds")}


def run(a):
    ds = datasets.load(a.dataset)
    ref = "--best" if a.best or not a.run else a.run
    rr = results.find(ds.name, ref)
    setup = results.setup_of(rr)
    chat, judge = llm.chat_name(a.chat), llm.judge_name(a.judge)
    for role, name in (("chat", chat), ("judge", judge)):
        try:
            models.no_paid_api(name, models.block(name, "chat")[1])
        except models.SetupProblem as e:
            sys.exit(f"! --{role} {name}: {e}")
    qs = spread(ds.questions, a.limit)
    print(f"ANSWERS {ds.name}: run {rr['phase']}/{rr['name']} ({setup.label()}, {rr['setup'].get('embed')})")
    print(f"  {len(qs)} questions; answers by {llm.describe(chat)}, judged by {llm.describe(judge)}"
          + ("  (the judge is the answer model: it may favour itself)" if chat == judge else ""))
    index = pipeline.build(ds, setup)
    system = pipeline.system_prompt(ds)
    scores = {x["id"]: x["scores"] for x in rr["questions"]}
    rows, t0 = [], time.time()
    print(f"\n   #  {'id':<14}{'context':>8}{'faithful':>9}{'answers':>8}{'must say':>9}  refusal")
    for q in qs:
        try:
            row = answer_one(ds, q, index, setup, chat, judge, system, scores.get(q.id) if q.gold else None)
        except models.SetupProblem as e:
            sys.exit(f"! {e}")
        rows.append(row)
        ref_ = ("refused ✓" if row["refusal_ok"] else "answered ✗ (should refuse)") if "refusal_ok" in row else \
            ("refused ✗" if row["refused"] else "")
        print(f"  {q.n:>2}  {q.id:<14}{results.fmt(row.get('context')):>8}{results.fmt(row.get('faithfulness')):>9}"
              f"{results.fmt(row.get('answer_relevance')):>8}{results.fmt(row.get('must_say')):>9}  {ref_}", flush=True)
    s = summarize(rows)
    print(f"\n  context relevance {results.fmt(s['context'])}   faithfulness {results.fmt(s['faithfulness'])}   "
          f"answer relevance {results.fmt(s['answer_relevance'])}"
          + (f"   must say {results.fmt(s['must_say'])}" if s["must_say"] is not None else "")
          + (f"   refusals right {sum(r.get('refusal_ok', False) for r in rows)} of {s['refusal_questions']}" if s["refusal_questions"] else ""))
    print("\n  The RAG triad:")
    print("  context relevance: did we send passages that fit? (Recall@5 from the run: the right passages among the 5 sent)")
    print("  faithfulness: is every claim in the answer backed by the passages sent? (the judge checks each claim)")
    print("  answer relevance: does the answer actually answer the question? (the judge rates it; 1.00 = fully)")
    if s["must_say"] is not None:
        print("  must say: share of the facts the golden set says the answer must contain")
    odd = [r for r in rows if r.get("context") is not None and (
        (r["context"] == 0 and (r.get("answer_relevance") or 0) >= 0.75) or (r["context"] >= 0.99 and (r.get("answer_relevance") or 1) <= 0.25))]
    for r in odd:
        kind = "answered well with none of the right passages: from the model's own knowledge?" if r["context"] == 0 else \
            "had every right passage but didn't answer"
        print(f"  ! question {r['n']} ({r['id']}) {kind}")
    out = {"dataset": ds.name, "phase": "answers", "name": f"{rr['phase']}-{rr['name']}__{chat}__{judge}", "run": f"{rr['phase']}/{rr['name']}",
           "setup": rr["setup"], "chat": chat, "chat_model": llm.describe(chat), "judge": judge, "judge_model": llm.describe(judge),
           "limit": a.limit, "rows": rows, "summary": s, "date": time.strftime("%Y-%m-%d %H:%M"), "took_s": round(time.time() - t0, 1)}
    p = results.path(ds.name, "answers", out["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    results.report()
    print(f"\n  saved: {p}")
    return 0


def table(ds_name):
    """The answers table for RESULTS.md."""
    import json
    files = sorted((results.root() / ds_name / "answers").glob("*.json"))
    if not files:
        return []
    lines = ["### answers (RAG triad)", "", "| Run | Chat | Judge | Q | Context rel. | Faithfulness | Answer rel. | Must say | Refusals right |",
             "|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        s = r["summary"]
        ok = sum(x.get("refusal_ok", False) for x in r["rows"])
        lines.append(f"| {r['run']} | {r['chat_model']} | {r['judge_model']} | {s['n']} | {results.fmt(s['context'])} | "
                     f"{results.fmt(s['faithfulness'])} | {results.fmt(s['answer_relevance'])} | {results.fmt(s['must_say'])} | "
                     + (f"{ok} of {s['refusal_questions']}" if s["refusal_questions"] else "—") + " |")
    return lines + [""]
