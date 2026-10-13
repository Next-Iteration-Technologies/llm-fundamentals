"""compare --phase P: one change at a time, every row scored on the whole golden set (experiments/P.toml).
Also `chunks` (what the chunkings look like, before any search) and `show` (gold next to what was found)."""
import statistics, sys, textwrap, time

from . import ROOT, chunkers, datasets, llm, metrics, pipeline, results

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def experiment(phase):
    p = ROOT / "experiments" / f"{phase}.toml"
    if not p.exists():
        sys.exit(f"! no experiment '{phase}': {p} is missing. Have: "
                 + ", ".join(sorted(x.stem for x in (ROOT / 'experiments').glob('*.toml'))))
    return tomllib.loads(p.read_text(encoding="utf-8"))


def rows_for(exp, ds):
    return [r for r in exp["rows"] if ds.name in r.get("only", [ds.name])]


def best_chunkings(ds, embed=None):
    """Chunkings ranked by their saved chunking-phase runs (best first)."""
    runs = [r for r in results.all_runs(ds.name) if r["phase"] == "chunking"]
    runs.sort(key=results.rank_key, reverse=True)
    return [r["setup"]["chunking"] for r in runs]


def resolve_chunkings(exp, ds):
    ranked = best_chunkings(ds)
    out = []
    for c in exp.get("chunkings", [ds.chunking]):
        if c.startswith("best-"):
            n = int(c.split("-")[1])
            if len(ranked) >= n:
                c = ranked[n - 1]
            else:
                print(f"! no saved chunking runs to pick '{c}' from (run compare --phase chunking first); using {ds.chunking}.")
                c = ds.chunking
        if c not in out:
            out.append(c)
    return out


def plan(a, ds, exp):
    """[(run name, setup, needs model)] for this phase and dataset."""
    out = []
    keep = exp.get("keep", {})
    if exp["phase"] == "chunking":
        for r in rows_for(exp, ds):
            out.append((r["name"], pipeline.Setup(chunking=r["name"], embed=a.embed or "", **keep), r.get("needs_model", False)))
    elif exp["phase"] == "retrieval":
        chunkings = resolve_chunkings(exp, ds)
        for i, ch in enumerate(chunkings):
            for r in rows_for(exp, ds):
                if r.get("main_only") and i:
                    continue
                s = pipeline.Setup(chunking=ch, retriever=r.get("retriever", "semantic"), rerank=r.get("rerank", False),
                                   transform=r.get("transform", ""), embed=a.embed or "", **keep)
                out.append((r["name"] if i == 0 else f"{r['name']}@{ch}", s, r.get("needs_model", False)))
    else:  # keep the best run of an earlier phase, change what the rows say
        src = [r for r in results.all_runs(ds.name) if r["phase"] == exp.get("from") and r["retriever"] == "pipeline"]
        base = results.setup_of(max(src, key=results.rank_key)) if src else pipeline.Setup(chunking=ds.chunking)
        if not src:
            print(f"! no saved {exp.get('from')} runs: keeping the starting setup ({base.label()}).")
        if base.transform:  # a rewrite needs a model on every question; the comparison keeps the search part
            base.transform = ""
        for r in rows_for(exp, ds):
            s = pipeline.Setup(**{**base.to_dict(), **{k: v for k, v in r.items() if k in pipeline.Setup.__dataclass_fields__}})
            if "embed" not in r:
                s.embed = a.embed or s.embed
            out.append((r["name"], s, r.get("needs_model", False)))
    if a.rows:
        keep_names = a.rows.split(",")
        out = [x for x in out if x[0] in keep_names or x[0].split("@")[0] in keep_names]
    if a.skip:
        skip = a.skip.split(",")
        out = [x for x in out if x[0] not in skip and x[0].split("@")[0] not in skip]
    return out


def compare(a):
    ds = datasets.load(a.dataset)
    exp = experiment(a.phase)
    if exp.get("only") and ds.name not in exp["only"]:
        sys.exit(f"! phase {a.phase} needs {' or '.join(exp['only'])} (dates, platforms, askers); add --dataset {exp['only'][0]}.")
    rows = plan(a, ds, exp)
    chat = None if a.dry else llm.chat_name(a.chat)
    print(f"COMPARE {exp['phase']} on {ds.name}: {len(rows)} setups, {len(ds.scored)} questions each. "
          f"One question moves an average by about {100 / max(1, len(ds.scored)):.0f}%.")
    if exp.get("question"):
        print(f"  {exp['question']}")
    print(f"\n  {'setup':<34}{'R@3':>6}{'R@5':>6}{'MRR':>6}{'nDCG':>6}{'all':>6}{'R@2k':>6}{'R@50':>6}   won / lost vs the row above")
    prev, done = None, []
    for order, (name, setup, needs_model) in enumerate(rows):
        if needs_model and a.dry:
            print(f"  {name:<34}  skipped: needs the chat model (run without --dry)")
            continue
        t0 = time.time()
        r = results.run(ds, setup, exp["phase"], name, dry=a.dry, chat=chat, limit=a.limit)
        r["order"] = order
        if needs_model and a.limit:
            r["note"] = f"model step on the first {a.limit} chunks only"
        results.save(r)
        s = r["summary"]
        line = f"  {name:<34}" + "".join(f"{results.fmt(s.get(k)):>6}" for k in ("R@3", "R@5", "MRR", "nDCG@5", "all@5", "R@2k", "R@50"))
        if prev:
            won, lost = results.changed(prev, r)
            line += f"   won {len(won)} {fmt_ids(won)} / lost {len(lost)} {fmt_ids(lost)}"
        print(line + f"   ({r['setup']['embed']}, {time.time() - t0:.0f} s)", flush=True)
        prev = r
        done.append(r)
    if done:
        b = max(done, key=results.rank_key)
        print(f"\n  best here: {b['name']} (Recall@5 {results.fmt(b['summary'].get('R@5'))}, MRR {results.fmt(b['summary'].get('MRR'))})")
        for k in ("R@3", "R@5", "MRR", "R@2k", "R@50"):
            print(f"  {metrics.EXPLAIN[k]}")
        print("  won / lost: the questions (by number) whose Recall@5 went up / down against the row above.")
        print(f"  saved in {results.root() / ds.name / exp['phase']}; all tables: python rag.py report")
    results.report()
    return 0


def fmt_ids(ids, n=8):
    return "(" + ", ".join(map(str, ids[:n])) + (", …" if len(ids) > n else "") + ")" if ids else ""


# ------------------------------------------------------------------ chunks
def cut_in_two(chunks, gold):
    """Gold passages that no single chunk holds whole: the cut went through them."""
    by_file = {}
    for c in chunks:
        by_file.setdefault(c["file"], []).append(c)
    return sum(1 for g in gold if not any(c["start"] <= g.start and g.end <= c["end"] for c in by_file.get(g.file, [])))


def chunks_cmd(a):
    ds = datasets.load(a.dataset)
    exp = experiment("chunking")
    names = [a.chunking] if a.chunking else [r["name"] for r in rows_for(exp, ds) if not r.get("needs_model")]
    emb = pipeline.embed_name(ds, a.embed)
    gold = [g for q in ds.questions for g in q.gold]
    print(f"CHUNKS {ds.name}: {len(ds.docs)} documents; {len(gold)} gold passages "
          f"(median {int(statistics.median([chunkers.ntok(ds.docs[g.file][g.start:g.end]) for g in gold]))} tokens)\n")
    print(f"  {'chunking':<14}{'chunks':>8}  {'tokens: min':>11}{'median':>8}{'max':>6}   gold passages cut in two")
    for n in names:
        chat = llm.chat_name(a.chat) if chunkers.parse(n)[0].__name__ in chunkers.NEEDS_CHAT else None
        cs = chunkers.chunk(ds, n, emb, chat, a.limit)
        sz = [chunkers.ntok(c["text"]) for c in cs]
        cut = cut_in_two(cs, gold)
        extra = f"   (searched as {len(chunkers.units(ds, n, emb, chat, a.limit)):,} small pieces)" if n.startswith("hierarchical") else ""
        print(f"  {n:<14}{len(cs):>8,}  {min(sz):>11}{int(statistics.median(sz)):>8}{max(sz):>6}   "
              f"{cut:>4} of {len(gold)} ({100 * cut / max(1, len(gold)):.0f}%){extra}")
    if a.show:
        for n in names:
            cs = chunkers.chunk(ds, n, emb, llm.chat_name(a.chat) if n.startswith("contextual") else None, a.limit)
            print(f"\n  {n}: {a.show} examples")
            picks = cs[:a.show] if a.limit else cs[:: max(1, len(cs) // a.show)][:a.show]  # --limit: the ones with context
            for c in picks:
                if c.get("context"):
                    print(f"    context: {c['context']}")
                print(textwrap.indent(textwrap.shorten(c["text"], 300), f"    [{c['source'][:50]}] "))
    print("\n  cut in two: no single chunk holds the whole passage, so the answer is split across chunks.")
    if not a.chunking:
        print("  Contextual chunking needs the chat model; see it with: python rag.py chunks --chunking contextual --limit 10 --show 5")


# ------------------------------------------------------------------ show
def mark(text, s, e, a, b):
    """text is [s, e) of a file; wrap the part that also lies in [a, b) in »…«."""
    lo, hi = max(s, a), min(e, b)
    if hi <= lo:
        return text
    return text[:lo - s] + "»" + text[lo - s:hi - s] + "«" + text[hi - s:]


def show_cmd(a):
    ds = datasets.load(a.dataset)
    q = ds.question(a.question)
    ref = a.run or ("eval/baseline" if results.path(ds.name, "eval", "baseline").exists() else None)
    r = results.find(ds.name, ref)
    row = next((x for x in r["questions"] if x["id"] == q.id), None)
    print(f"SHOW question {q.n} ({q.id}) in run {r['phase']}/{r['name']} ({results.setup_of(r).label()}; {r['setup'].get('embed')})")
    print(f"  {q.text}")
    if row is None:
        print("  This question has no gold passages, so it isn't scored.")
        return 0
    s = row["scores"]
    print(f"  Recall@3 {results.fmt(s.get('R@3'))}, Recall@5 {results.fmt(s.get('R@5'))}, MRR {results.fmt(s.get('MRR'))}\n")
    top = row["top"]
    for i, ((f, gs, ge, grade), rank) in enumerate(zip(row["gold"], row["ranks"]), 1):
        gold_text = ds.docs[f][gs:ge]
        found = next(((n, t) for n, t in enumerate(top, 1) if t[0] == f and min(t[2], ge) - max(t[1], gs) > 0), None)
        state = f"found at rank {rank}" if rank else "not found in the top 50"
        print(f"  GOLD {i} of {len(row['gold'])}  [{f} {gs}-{ge}]  {state}")
        if found:
            n, (cf, cs, ce, sc) = found
            held = (min(ce, ge) - max(cs, gs)) / (ge - gs)
            print(textwrap.indent(textwrap.fill(mark(gold_text, gs, ge, cs, ce), 100), "    | "))
            print(f"  FOUND rank {n}: chunk [{cf} {cs}-{ce}] holds {100 * held:.0f}% of this passage"
                  + (" (the cut went through it)" if held < 1 else ""))
            print(textwrap.indent(textwrap.fill(mark(ds.docs[cf][cs:ce], cs, ce, gs, ge), 100), "    | "))
        else:
            print(textwrap.indent(textwrap.fill(gold_text, 100), "    | "))
            f1, s1, e1, _ = top[0]
            print(f"  No chunk in the top 10 touches it. Rank 1 was [{f1} {s1}-{e1}]:")
            print(textwrap.indent(textwrap.shorten(ds.docs[f1][s1:e1], 400), "    | "))
        print()
    print("  »…« marks the text that the gold passage and the chunk share.")
    return 0
