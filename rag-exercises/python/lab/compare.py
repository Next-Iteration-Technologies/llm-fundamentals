"""compare --phase P: one change at a time, every row scored on the whole golden set (experiments/P.toml).
Also `chunks` (what the chunkings look like, before any search) and `show` (gold next to what was found)."""
import statistics, sys, textwrap, time

import models

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
        chunkings = a.chunkings.split(",") if a.chunkings else resolve_chunkings(exp, ds)
        for i, ch in enumerate(chunkings):
            for r in rows_for(exp, ds):
                if r.get("main_only") and i:
                    continue
                s = pipeline.Setup(chunking=ch, retriever=r.get("retriever", "semantic"), rerank=r.get("rerank", False),
                                   transform=r.get("transform", ""), embed=a.embed or "", **keep)
                out.append((r["name"] if i == 0 else f"{r['name']}@{ch}", s, r.get("needs_model", False)))
    else:  # keep the best run of an earlier phase, change what the rows say
        froms = exp.get("from", [])
        froms = [froms] if isinstance(froms, str) else froms
        src = []
        for ph in froms:  # the first earlier phase that has runs
            src = src or [r for r in results.all_runs(ds.name) if r["phase"] == ph and r["retriever"] == "pipeline"]
        if a.setup:
            src = [results.find(ds.name, a.setup)]
        base = results.setup_of(max(src, key=results.rank_key)) if src else pipeline.Setup(chunking=ds.chunking)
        if not src:
            print(f"! no saved {' or '.join(froms)} runs: keeping the starting setup ({base.label()}).")
        else:
            b = max(src, key=results.rank_key)
            print(f"  keeping the best setup so far: {b['phase']}/{b['name']} ({base.label()}, {b['setup'].get('embed')})")
        if base.transform:  # a rewrite needs a model on every question; the comparison keeps the search part
            base.transform = ""
        for r in rows_for(exp, ds):
            s = pipeline.Setup(**{**base.to_dict(), **{k: v for k, v in r.items() if k in pipeline.Setup.__dataclass_fields__}})
            if "embed" not in r:
                s.embed = a.embed or s.embed
            out.append((r["name"], s, r.get("needs_model", False)))
    out = [(n, st, nm, i) for i, (n, st, nm) in enumerate(out)]   # the order in the experiment, whatever is left out
    asked = a.rows.split(",") if a.rows else []
    on_request = {r["name"] for r in exp["rows"] if r.get("on_request")}
    for name, *_ in out:
        if name.split("@")[0] in on_request and name not in asked:
            print(f"  ({name} is left out: it needs a model call per {'chunk' if exp['phase'] == 'chunking' else 'question'}. "
                  f"Add it with --rows {name})")
    out = [x for x in out if x[0].split("@")[0] not in on_request or x[0].split("@")[0] in asked]
    if asked:
        out = [x for x in out if x[0] in asked or x[0].split("@")[0] in asked]
    if a.skip:
        skip = a.skip.split(",")
        out = [x for x in out if x[0] not in skip and x[0].split("@")[0] not in skip]
    return [x for x in out if x[0] in asked or embed_answers(exp, x[0].split("@")[0])]


_answers = {}


def embed_answers(exp, row):
    """A row whose embed block this laptop can't reach (no Ollama, no Nexus) is left out, with the reason.
    Named in --rows, it runs anyway and fails loudly."""
    embed = next((r.get("embed") for r in exp["rows"] if r["name"] == row), None)
    if not embed or embed == "hash":
        return True
    if embed not in _answers:
        if not isinstance(models.PROVIDERS.get(embed), dict):
            _answers[embed] = (False, f"no [{embed}] block in providers.toml (python setup.py adds it)")
        else:
            ok, why = models.reachable(embed, "embed")
            if not ok and models.PROVIDERS[embed].get("kind") == "ollama" and "pull" not in why:
                why = f"Ollama doesn't answer at {models.PROVIDERS[embed].get('base_url')}"
            _answers[embed] = (ok, why)
    ok, why = _answers[embed]
    if not ok:
        print(f"  ({row} is left out: {why.rstrip('.')}. Add it with --rows {row} once it answers)")
    return ok


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
    prev, done, group = None, [], None
    grouped = len({x[1].chunking for x in rows}) > 1 and exp["phase"] != "chunking"
    for name, setup, needs_model, order in rows:
        if grouped and setup.chunking != group:
            group, prev = setup.chunking, None
            print(f"  -- chunking: {group} --")
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
        for w in exp.get("watch", []):
            print("      " + watch_line(r, w, ds))
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


def watch_line(r, w, ds):
    """Which of two disagreeing documents comes first for one question, in this run's top 10."""
    row = next((x for x in r["questions"] if x["id"] == w["question"]), None)
    if row is None:
        return f"{w['question']}: not in this run"
    rank = {}
    for n, (f, *_rest) in enumerate(row["top"], 1):
        rank.setdefault(f, n)
    ro, rn = rank.get(w["old"]), rank.get(w["new"])
    name = lambda f: f.split("/")[-1]
    if not ro and not rn:
        first = "neither in the top 10"
    elif rn and (not ro or rn < ro):
        first = f"the newer one first: {name(w['new'])} rank {rn}" + (f", {name(w['old'])} rank {ro}" if ro else f", {name(w['old'])} not in the top 10")
    else:
        first = f"the older one first: {name(w['old'])} rank {ro}" + (f", {name(w['new'])} rank {rn}" if rn else f", {name(w['new'])} not in the top 10")
    return f"question {ds.question(w['question']).n} ({w['question']}), {w['say']} → {first}"


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


def box(text, width=100, most=700):
    """Indented, wrapped, and cut after `most` characters (a gold passage can be thousands long)."""
    t = " ".join(text.split())
    if len(t) > most:
        t = t[:most] + " […]" + ("«" if t[:most].count("»") > t[:most].count("«") else "")
    return textwrap.indent(textwrap.fill(t, width), "    | ")


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
    untouched = 0
    for i, ((f, gs, ge, grade), rank) in enumerate(zip(row["gold"], row["ranks"]), 1):
        gold_text = ds.docs[f][gs:ge]
        found = next(((n, t) for n, t in enumerate(top, 1) if t[0] == f and min(t[2], ge) - max(t[1], gs) > 0), None)
        state = f"found at rank {rank}" if rank else "not found in the top 50"
        print(f"  GOLD {i} of {len(row['gold'])}  [{f} {gs}-{ge}]  {state}")
        if found:
            n, (cf, cs, ce, sc) = found
            held = (min(ce, ge) - max(cs, gs)) / (ge - gs)
            print(box(mark(gold_text, gs, ge, cs, ce)))
            print(f"  FOUND rank {n}: chunk [{cf} {cs}-{ce}] holds {100 * held:.0f}% of this passage"
                  + (" (the cut went through it)" if held < 1 else ""))
            print(box(mark(ds.docs[cf][cs:ce], cs, ce, gs, ge)))
        else:
            untouched += 1
            print(box(gold_text))
            print("  No chunk in the top 10 touches it.")
        print()
    if untouched:
        f1, s1, e1, _ = top[0]
        print(f"  For comparison, the chunk at rank 1 was [{f1} {s1}-{e1}]:")
        print(box(ds.docs[f1][s1:e1]))
        print()
    print("  »…« marks the text that the gold passage and the chunk share.")
    return 0
