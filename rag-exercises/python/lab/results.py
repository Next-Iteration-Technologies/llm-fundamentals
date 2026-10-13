"""Runs: one setup, every golden question, scores saved. RESULTS.md is rebuilt from the run files, never edited by hand.

python/results/<dataset>/<phase>/<run>.json  (RAG_RESULTS moves the folder; selftest uses a scratch one)
Run the same setup twice, get the same file, apart from `date` and `took_s`.
"""
import json, os, random, re, sys, time
from pathlib import Path

from . import PY, chunkers, datasets, metrics, pipeline

VOLATILE = ("date", "took_s")   # the only fields allowed to differ between two runs of the same setup
NOT_SETUPS = ("perfect", "random")


def root():
    return Path(os.environ.get("RAG_RESULTS") or PY / "results")


def path(ds_name, phase, name):
    return root() / ds_name / phase / f"{re.sub(r'[^A-Za-z0-9_.@+-]', '_', name)}.json"


# ------------------------------------------------------------------ one run
def run(ds, setup, phase, name, retriever="pipeline", dry=False, chat=None, limit=None, index=None, quiet=False):
    """Score every question with gold. retriever: pipeline (the setup), perfect (the gold itself) or random."""
    t0 = time.time()
    index = index or (pipeline.build(ds, setup, dry, chat, limit) if retriever != "perfect" else None)
    rows = []
    for q in ds.scored:
        if retriever == "perfect":  # retrieve exactly the gold passages, the most important first
            order = sorted(q.gold, key=lambda g: -g.grade)
            ranked = [{"file": g.file, "start": g.start, "end": g.end, "text": ds.docs[g.file][g.start:g.end]} for g in order]
            scores = [1.0] * len(ranked)
        elif retriever == "random":
            rng = random.Random(f"{ds.name}|{q.id}")
            ranked = rng.sample(index.chunks, min(pipeline.DEPTH, len(index.chunks)))
            scores = [0.0] * len(ranked)
        else:
            final, _ = pipeline.retrieve(index, q.text, setup, queries=transform_queries(q.text, setup, chat, dry))
            ranked, scores = [c for c, _ in final], [s for _, s in final]
        ranks = metrics.gold_ranks(ranked, q.gold)
        sc = metrics.score(ranks, [g.grade for g in q.gold], [chunkers.ntok(c["text"]) for c in ranked])
        rows.append({"n": q.n, "id": q.id, "level": q.level, "ranks": ranks,
                     "gold": [[g.file, g.start, g.end, g.grade] for g in q.gold],
                     "top": [[c["file"], c["start"], c["end"], round(s, 4)] for c, s in zip(ranked[:10], scores[:10])],
                     "scores": {k: round(v, 4) for k, v in sc.items()}})
    out = {"dataset": ds.name, "phase": phase, "name": name, "retriever": retriever,
           "setup": dict(setup.to_dict(), chunking=setup.chunking or ds.chunking,
                         embed=index.embed if index else "", embedder=pipeline.embed_desc(index.embed) if index else ""),
           "chunks": len(index.chunks) if index else 0, "questions": rows, "summary": summarize(rows),
           "date": time.strftime("%Y-%m-%d %H:%M"), "took_s": round(time.time() - t0, 1)}
    if chat and (setup.transform or (setup.chunking or "").startswith("contextual")):
        out["setup"]["chat"] = chat
        if limit and (setup.chunking or "").startswith("contextual"):
            out["setup"]["context_limit"] = limit
    return out


def transform_queries(question, setup, chat, dry):
    if not setup.transform:
        return None
    from . import retrievers
    return retrievers.transform(question, setup.transform, chat, dry)


def summarize(rows):
    s = {k: round(metrics.mean([r["scores"] for r in rows], k), 4) for k in metrics.NAMES
         if metrics.mean([r["scores"] for r in rows], k) is not None}
    s["n"] = len(rows)
    levels = sorted({r["level"] for r in rows if r["level"]})
    if len(levels) > 1:
        s["by_level"] = {lv: {k: round(metrics.mean([r["scores"] for r in rows if r["level"] == lv], k), 4)
                              for k in ("R@3", "R@5", "MRR", "nDCG@5")} | {"n": sum(r["level"] == lv for r in rows)}
                         for lv in levels}
    return s


def save(r):
    p = path(r["dataset"], r["phase"], r["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


def stable(r):
    """A run without the fields that may change between two identical runs."""
    return {k: v for k, v in r.items() if k not in VOLATILE}


# ------------------------------------------------------------------ finding runs
def all_runs(ds_name):
    base = root() / ds_name
    out = []
    for p in sorted(base.glob("*/*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return [r for r in out if "questions" in r and "summary" in r]


def rank_key(r):
    s = r["summary"]
    return (s.get("R@5", 0), s.get("MRR", 0), s.get("nDCG@5", 0))


def best(ds_name):
    """The best real setup so far: highest Recall@5, then MRR (perfect and random don't count)."""
    runs = [r for r in all_runs(ds_name) if r.get("retriever") == "pipeline"]
    if not runs:
        sys.exit(f"! no saved runs for {ds_name} yet. Run `python rag.py eval` first (Chapter 2).")
    return max(runs, key=rank_key)


def find(ds_name, ref):
    """A run by 'phase/name', by name alone, or --best."""
    if ref in (None, "", "--best", "best"):
        return best(ds_name)
    if "/" in ref:
        phase, name = ref.split("/", 1)
        p = path(ds_name, phase, name)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    hits = [r for r in all_runs(ds_name) if r["name"] == ref or f"{r['phase']}/{r['name']}" == ref]
    if len(hits) == 1:
        return hits[0]
    have = ", ".join(f"{r['phase']}/{r['name']}" for r in all_runs(ds_name)) or "none yet"
    sys.exit(f"! {'more than one' if hits else 'no'} run '{ref}' for {ds_name}. Have: {have}")


def setup_of(r):
    s = dict(r["setup"])
    return pipeline.Setup(**{k: s[k] for k in ("chunking", "retriever", "rerank", "embed", "k", "transform", "filters") if k in s})


# ------------------------------------------------------------------ printing
def fmt(v):
    return "  -  " if v is None else f"{v:.2f}"


def print_run(r, ds, show_questions=True):
    s = r["summary"]
    setup = r["setup"]
    what = {"perfect": "PERFECT student: retrieves exactly the gold passages",
            "random": "RANDOM guesser: 50 random chunks per question"}.get(r["retriever"], "")
    levels = s.get("by_level", {})
    lv = " (" + ", ".join(f"{k} {v['n']}" for k, v in levels.items()) + ")" if levels else ""
    print(f"EVAL {ds.name}: {s['n']} questions with gold passages{lv}")
    print(f"  {what or 'setup: ' + pipeline.Setup(**{k: setup[k] for k in ('chunking', 'retriever', 'rerank', 'transform') if k in setup}).label()}"
          + (f"; embedding {setup.get('embedder')}" if r["retriever"] != "perfect" else ""))
    if show_questions:
        print(f"\n   #  {'id':<14} R@3   R@5   MRR   question")
        for row in r["questions"]:
            q = ds.question(row["id"])
            miss = "  " if row["scores"].get("R@5") == 1 else "✗ "
            print(f"  {row['n']:>2}  {row['id']:<14}{fmt(row['scores'].get('R@3'))}  {fmt(row['scores'].get('R@5'))}  "
                  f"{fmt(row['scores'].get('MRR'))}  {miss}{q.text[:60]}")
    print(f"\n  Recall@3 {fmt(s.get('R@3'))}   Recall@5 {fmt(s.get('R@5'))}   MRR {fmt(s.get('MRR'))}   "
          f"nDCG@5 {fmt(s.get('nDCG@5'))}   all found {fmt(s.get('all@5'))}   Recall@~2k tokens {fmt(s.get('R@2k'))}")
    for name, v in levels.items():
        print(f"    {name:<8} ({v['n']}): Recall@3 {fmt(v.get('R@3'))}   Recall@5 {fmt(v.get('R@5'))}   MRR {fmt(v.get('MRR'))}")
    print()
    for k in ("R@3", "MRR", "R@5", "all@5"):
        print(f"  {metrics.EXPLAIN[k]}")
    if show_questions and r["retriever"] == "pipeline":
        missed = [row["n"] for row in r["questions"] if row["scores"].get("R@5", 1) < 1]
        if missed:
            print(f"\n  ✗ = a right passage is missing from the top 5: {len(missed)} questions ({', '.join(map(str, missed[:15]))}"
                  f"{', …' if len(missed) > 15 else ''}). Look at one: python rag.py show {missed[0]}")
    print(f"\n  One question moves the average by about {100 / max(1, s['n']):.0f}%.")


def changed(prev, cur, key="R@5"):
    """Which questions this run won and lost against the one before: per question, did `key` go up or down?"""
    a = {r["id"]: r["scores"].get(key, 0) for r in prev["questions"]}
    won = [r["n"] for r in cur["questions"] if r["scores"].get(key, 0) > a.get(r["id"], 0) + 1e-9]
    lost = [r["n"] for r in cur["questions"] if r["scores"].get(key, 0) < a.get(r["id"], 0) - 1e-9]
    return won, lost


# ------------------------------------------------------------------ RESULTS.md
COLS = [("R@3", "R@3"), ("R@5", "R@5"), ("MRR", "MRR"), ("nDCG@5", "nDCG@5"), ("all@5", "All found"), ("R@2k", "R@~2k tok"),
        ("R@50", "R@50")]
PHASES = ["eval", "chunking", "retrieval", "embedding", "filters"]


def report():
    lines = ["# Results", "", "*Made by `python rag.py report` from the run files in this folder. Don't edit by hand.*", "",
             "Scores are per question, then averaged. R@k = Recall@k (of the right passages, how many in the top k); "
             "MRR = how high the first right chunk came; All found = every right passage in the top 5; "
             "R@~2k tok = Recall when every setup sends about 2,000 tokens; R@50 = in the top 50 at all.", ""]
    for d in sorted(p.name for p in root().glob("*") if p.is_dir()):
        runs = all_runs(d)
        if not runs:
            continue
        lines += [f"## {d}", ""]
        phases = PHASES + sorted({r["phase"] for r in runs} - set(PHASES))
        for ph in phases:
            rows = [r for r in runs if r["phase"] == ph]
            if not rows:
                continue
            lines += [f"### {ph}", "", "| Run | Chunking | Search | Embedding | Q | " + " | ".join(c for _, c in COLS) + " | Changed vs row above |",
                      "|---|---|---|---|---|" + "---|" * len(COLS) + "---|"]
            order = sorted(rows, key=lambda r: r.get("order", 0)) if any("order" in r for r in rows) else rows
            prev = None
            for r in order:
                st = r["setup"]
                search = "—" if r["retriever"] != "pipeline" else st["retriever"] + (" + rerank" if st.get("rerank") else "") \
                    + (f" + {st['transform']}" if st.get("transform") else "") + "".join(f", {k}={v}" for k, v in sorted((st.get("filters") or {}).items()))
                chg = ""
                if prev is not None and r["retriever"] == "pipeline" and prev["retriever"] == "pipeline":
                    won, lost = changed(prev, r)
                    chg = f"won {len(won)}, lost {len(lost)}"
                lines.append(f"| {r['name']} | {st.get('chunking', '')} | {search} | {st.get('embed') or '—'} | {r['summary']['n']} | "
                             + " | ".join(fmt(r["summary"].get(k)) for k, _ in COLS) + f" | {chg} |")
                prev = r
            levels = sorted({lv for r in order for lv in r["summary"].get("by_level", {})})
            if levels:
                lines += ["", "By level (R@3 / R@5 / MRR):", "", "| Run | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
                for r in order:
                    bl = r["summary"].get("by_level", {})
                    lines.append(f"| {r['name']} | " + " | ".join(
                        f"{fmt(bl.get(lv, {}).get('R@3'))} / {fmt(bl.get(lv, {}).get('R@5'))} / {fmt(bl.get(lv, {}).get('MRR'))}" for lv in levels) + " |")
            lines.append("")
        lines += extra_tables(d)
    out = root() / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def extra_tables(d):
    """Answer runs and the final comparison add their own tables (phases 6 and 8)."""
    lines = []
    for mod in ("answers", "final"):
        try:
            m = __import__(f"lab.{mod}", fromlist=["table"])
        except ImportError:
            continue
        if hasattr(m, "table"):
            lines += m.table(d)
    return lines


def walkthrough_compare(a, ds, q, text, setup):
    """Chapter 10: the same question with the best setup and with RUN's setup, one after the other."""
    from . import walkthrough
    other = find(ds.name, a.compare_with)
    top = best(ds.name)
    for title, r in ((f"  [best setup: {top['phase']}/{top['name']}]", top), (f"  [{other['phase']}/{other['name']}]", other)):
        s = setup_of(r)
        s.embed = a.embed or s.embed
        print("=" * 100)
        walkthrough.walk(a, ds, q, text, s, title=title)
    print("=" * 100)
    print(f"Compared: best setup {top['phase']}/{top['name']} ({setup_of(top).label()}) "
          f"against {other['phase']}/{other['name']} ({setup_of(other).label()}).")
