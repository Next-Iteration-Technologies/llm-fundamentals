"""python rag.py final: RAG vs the whole corpus in the prompt, same model, same questions, one judge (phase 8).

Two rows per model (--model, default the chat model):
  rag    the best RAG setup's chunks and the question
  full   every document and the question: no chunks, no search
Both are scored the same way: answer relevance, must say, refusals, citations, tokens and seconds per question.
Faithfulness is judged against the gold passages for both rows (the judge can't read the whole corpus for every
claim), so it reads "is the answer backed by the right passages", the same yardstick for both.
Anthropic blocks send the corpus as a cached system prompt: written once, read cheaply after.
Before anything is sent, the tokens each row will send are printed, and it waits for y (--yes skips that).
Contoso: what the model may not read (E9) is held back in both rows; --ignore-data-class lifts that for this
fictional pack, and the run is marked so.
"""
import json, re, sys, time

import models

from . import chunkers, datasets, llm, pipeline, results
from .answers import REFUSED, judge_faith, judge_relevance, says, spread


CHARS_PER_TOKEN = 2.4   # measured: the D&D full-context prompt, about 1.5 million characters, is 627,656 Claude tokens


def corpus_passages(ds):
    """Every document, as the model gets it with full context: Contoso as rag.py reads it (so data classes are known),
    other datasets one passage per file."""
    if ds.cfg.get("chunking") == "structure":
        return chunkers.chunk(ds, "structure")
    meta = chunkers.file_meta(ds)
    return [dict(meta.get(f, {}), file=f, start=0, end=len(t), text=t, source=f) for f, t in ds.docs.items()]


def full_system(ds, passages):
    return pipeline.system_prompt(ds) + "\n\nDOCUMENTS\n" + "\n".join(f"[{p['source']}] {p['text']}" for p in passages)


def run(a):
    ds = datasets.load(a.dataset)
    model = llm.chat_name(a.model)
    judge = llm.judge_name(a.judge)
    for role, name in (("model", model), ("judge", judge)):
        try:
            models.no_paid_api(name, models.block(name, "chat")[1])
        except models.SetupProblem as e:
            sys.exit(f"! --{role} {name}: {e}")
    rr = results.find(ds.name, a.setup or "--best")
    setup = results.setup_of(rr)
    qs = spread(ds.questions, a.limit)
    where = "local" if a.ignore_data_class else models.where(model)
    may = models.MAY_READ[where]
    index = pipeline.build(ds, setup)
    system = pipeline.system_prompt(ds)

    # what each row will send
    plan = []
    corpus = corpus_passages(ds)
    corpus_ok = [p for p in corpus if p.get("cls", "Public") in may]
    fsys = full_system(ds, corpus_ok)
    for q in qs:
        final, _ = pipeline.retrieve(index, q.text, setup, q=q)
        sent = [c for c, _ in final[:setup.k] if c.get("cls", "Public") in may]
        plan.append((q, sent))
    rag_tok = [chunkers.ntok(system) + chunkers.ntok(pipeline.build_prompt(q.text, sent)) for q, sent in plan]
    full_tok = chunkers.ntok(fsys)
    kind = models.block(model, "chat")[1]["kind"]
    print(f"FINAL {ds.name}: RAG vs full context, model {llm.describe(model)}, judge {llm.describe(judge)}")
    print(f"  RAG setup: {rr['phase']}/{rr['name']} ({setup.label()}, {rr['setup'].get('embed')}); {len(qs)} questions")
    if len(corpus_ok) < len(corpus):
        print(f"  data class: a '{where}' model may read {', '.join(sorted(c for c in may if c))}: "
              f"{len(corpus) - len(corpus_ok)} of {len(corpus)} passages held back in both rows (E9). --ignore-data-class lifts it.")
    elif a.ignore_data_class:
        print("  --ignore-data-class: nothing held back (fictional pack only); the run is marked so.")
    full_hi = int(len(fsys) / CHARS_PER_TOKEN)
    rag_hi = [int((len(system) + len(pipeline.build_prompt(q.text, sent))) / CHARS_PER_TOKEN) for q, sent in plan]
    print(f"\n  tokens to send (about):     per question                 all {len(qs)} questions")
    print(f"    RAG                      {sum(rag_tok) // max(1, len(rag_tok)):>10,} to {sum(rag_hi) // max(1, len(rag_hi)):>9,}   "
          f"{sum(rag_tok):>12,} to {sum(rag_hi):>11,}")
    print(f"    full context             {full_tok:>10,} to {full_hi:>9,}   {full_tok * len(qs):>12,} to {full_hi * len(qs):>11,}"
          + ("\n    (cached after the first question: later ones read it at a fraction of the price)" if kind == "anthropic" else ""))
    print("    The low figure counts words and punctuation; the high one, characters / 2.4, is closer to what Claude counts")
    print("    (the D&D rulebook measured 627,656 Claude tokens).")
    print("  Check current prices before a big run. The judge adds two short calls per answer.")
    if not a.yes:
        try:
            ok = input("  Send? [y/N] ").strip().lower() == "y"
        except EOFError:  # nothing to read the answer from
            sys.exit("\n! nothing sent: answer y, or add --yes.")
        if not ok:
            sys.exit("  nothing sent.")

    rows = {"rag": [], "full": []}
    t_all = time.time()
    print(f"\n   #  {'id':<14}{'row':<6}{'answers':>8}{'faithful':>9}{'must say':>9}{'tokens in':>11}{'seconds':>9}  note")
    for q, sent in plan:
        gold = [{"source": f"{g.file} {g.start}-{g.end}", "text": ds.docs[g.file][g.start:g.end]} for g in q.gold]
        for row_name, sys_prompt, prompt, cache_sys in (
                ("rag", system, pipeline.build_prompt(q.text, sent), False),
                ("full", fsys, f"QUESTION\n{q.text}", True)):
            row = {"n": q.n, "id": q.id, "level": q.level}
            try:
                r = llm.ask(model, sys_prompt, prompt, cache_system=cache_sys)
            except models.PromptTooBig as e:
                row.update(error="too big for the window", message=str(e), tokens_in=e.tokens)
                rows[row_name].append(row)
                print(f"  {q.n:>2}  {q.id:<14}{row_name:<6}{'—':>8}{'—':>9}{'—':>9}{e.tokens or 0:>11,}{'—':>9}  too big for the window")
                continue
            except models.SetupProblem as e:
                sys.exit(f"! {e}")
            ans = r["text"]
            u = r.get("usage") or {}
            row.update(answer=ans, seconds=r["seconds"], saved=bool(r["saved"]),
                       tokens_in=(u.get("input_tokens") or 0) + (u.get("cache_read_tokens") or 0) + (u.get("cache_write_tokens") or 0)
                       or chunkers.ntok(sys_prompt) + chunkers.ntok(prompt),
                       cache_read=u.get("cache_read_tokens"), cache_write=u.get("cache_write_tokens"))
            row["refused"] = bool(REFUSED.search(ans))
            if q.refuse:
                row["refusal_ok"] = row["refused"]
            if q.must_say:
                row["must_say"] = round(sum(says(ans, p) for p in q.must_say) / len(q.must_say), 4)
            cites = re.findall(r"\[([^\]]+)\]", ans)
            row["citations"] = len(cites)
            if q.gold:
                row["cites_right_file"] = any(any(g.file.split("/")[-1].split(".")[0] in c for g in q.gold) for c in cites)
                row["faithfulness"], _ = judge_faith(judge, gold, ans)
            row["answer_relevance"], _ = judge_relevance(judge, q.text, ans)
            rows[row_name].append(row)
            print(f"  {q.n:>2}  {q.id:<14}{row_name:<6}{results.fmt(row.get('answer_relevance')):>8}{results.fmt(row.get('faithfulness')):>9}"
                  f"{results.fmt(row.get('must_say')):>9}{row['tokens_in']:>11,}{row['seconds']:>9}"
                  + ("  saved reply" if row["saved"] else ""), flush=True)
    summary = {k: summarize(v) for k, v in rows.items()}
    print()
    for k, s in summary.items():
        print(f"  {k:<5} answered {s['answered']} of {s['n']}"
              + (f", {s['too_big']} too big for the window" if s["too_big"] else "")
              + f"; answer relevance {results.fmt(s['answer_relevance'])}, faithfulness to the right passages "
                f"{results.fmt(s['faithfulness'])}, tokens per question {s['tokens_in'] or 0:,.0f}, seconds {results.fmt(s['seconds'])}")
    out = {"dataset": ds.name, "phase": "final", "name": f"{model}__{rr['phase']}-{rr['name']}", "model": model,
           "model_desc": llm.describe(model), "judge": judge, "judge_desc": llm.describe(judge), "setup_run": f"{rr['phase']}/{rr['name']}",
           "setup": rr["setup"], "ignore_data_class": bool(a.ignore_data_class), "limit": a.limit, "rows": rows,
           "summary": summary, "corpus_tokens": full_tok, "date": time.strftime("%Y-%m-%d %H:%M"), "took_s": round(time.time() - t_all, 1)}
    p = results.path(ds.name, "final", out["name"] + ("__all-classes" if a.ignore_data_class else ""))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    results.report()
    print(f"\n  saved: {p}")
    return 0


def summarize(rows):
    ok = [r for r in rows if "error" not in r]

    def avg(k, src=ok):
        v = [float(r[k]) for r in src if r.get(k) is not None]
        return round(sum(v) / len(v), 4) if v else None
    return {"n": len(rows), "answered": len(ok), "too_big": sum(1 for r in rows if r.get("error") == "too big for the window"),
            "answer_relevance": avg("answer_relevance"), "faithfulness": avg("faithfulness"), "must_say": avg("must_say"),
            "refusals_ok": avg("refusal_ok"), "citations": avg("citations"), "cites_right_file": avg("cites_right_file"),
            "tokens_in": avg("tokens_in", rows), "seconds": avg("seconds")}


def table(ds_name):
    files = sorted((results.root() / ds_name / "final").glob("*.json"))
    if not files:
        return []
    lines = ["### final: RAG vs full context (same model, same judge)", "",
             "| Model | Row | Answered | Answer rel. | Faithful to right passages | Must say | Refusals right | Cites right file | Tokens / question | Seconds / question | Judge |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        for k in ("rag", "full"):
            s = r["summary"][k]
            note = f" ({s['too_big']} too big for the window)" if s["too_big"] else ""
            lines.append(f"| {r['model_desc']}{' (all classes)' if r.get('ignore_data_class') else ''} | {'RAG' if k == 'rag' else 'full context'} | "
                         f"{s['answered']} of {s['n']}{note} | {results.fmt(s['answer_relevance'])} | {results.fmt(s['faithfulness'])} | "
                         f"{results.fmt(s['must_say'])} | {results.fmt(s['refusals_ok'])} | {results.fmt(s['cites_right_file'])} | "
                         f"{(s['tokens_in'] or 0):,.0f} | {results.fmt(s['seconds'])} | {r['judge_desc']} |")
    return lines + [f"RAG setup: {r['setup_run']}.", ""]
