"""python rag.py misses RUN|--best: why does it still miss? (phase 7)

A miss is a question where a gold passage is not among the chunks sent (the top k). Each miss gets exactly one label,
the first of these that fits:
  cut apart          a missing passage: no chunk of this chunking holds half of it        -> chapter 3, chunking
  half found         a multi-part answer: some passages sent, others not                   -> chapter 3 or 4
  ranked too low     a missing passage was found, but below the cut-off (rank k+1 to 50)    -> chapter 4, retrieval
  wrong document     nothing in the top 50, and every chunk sent came from another file    -> chapter 5, embedding
  not found at all   nothing in the top 50, though some chunk sent was from the right file -> chapter 4 or 5
"""
from . import chunkers, datasets, results

LABELS = [("cut apart", "Chapter 3: cutting"), ("half found", "Chapter 3 or 4: hierarchical, more chunks"),
          ("ranked too low", "Chapter 4: searching"), ("wrong document", "Chapter 5: the embedding model"),
          ("not found at all", "Chapter 4 or 5")]


def held(chunks_by_file, f, s, e):
    """The largest share of the passage [s, e) that one chunk holds."""
    return max((max(0, min(c["end"], e) - max(c["start"], s)) / (e - s) for c in chunks_by_file.get(f, [])), default=0)


def label(row, k, chunks_by_file):
    """(label, detail) for one question of a run, or None if every passage was sent."""
    gold, ranks = row["gold"], row["ranks"]
    missing = [(i, g, r) for i, (g, r) in enumerate(zip(gold, ranks), 1) if not r or r > k]
    found_sent = [r for r in ranks if r and r <= k]
    if not missing or len(found_sent) >= min(k, len(gold)):
        return None
    for i, (f, s, e, _), r in missing:
        h = held(chunks_by_file, f, s, e)
        if h < 0.5:
            return "cut apart", f"no chunk holds half of passage {i} (at most {100 * h:.0f}%)"
    if found_sent:
        return "half found", f"{len(found_sent)} of {len(gold)} passages sent; " + \
            ", ".join(f"passage {i} at rank {r}" if r else f"passage {i} not in the top 50" for i, _, r in missing[:3])
    low = [(i, r) for i, _, r in missing if r]
    if low:
        i, r = min(low, key=lambda x: x[1])
        return "ranked too low", f"passage {i} first found at rank {r}, but only {k} chunks are sent"
    gold_files = {g[0] for g in gold}
    if not any(t[0] in gold_files for t in row["top"][:k]):
        return "wrong document", "every chunk sent came from " + ", ".join(sorted({t[0].split('/')[-1] for t in row["top"][:k]}))
    return "not found at all", "no right passage in the top 50"


def run(a):
    ds = datasets.load(a.dataset)
    r = results.find(ds.name, "--best" if a.best or not a.run else a.run)
    setup = results.setup_of(r)
    emb = r["setup"].get("embed") or None
    chunks = chunkers.chunk(ds, setup.chunking, emb, r["setup"].get("chat"), r["setup"].get("context_limit"))
    by_file = {}
    for c in chunks:
        by_file.setdefault(c["file"], []).append(c)
    k = setup.k
    rows = [(row, label(row, k, by_file)) for row in r["questions"]]
    missed = [(row, lab) for row, lab in rows if lab]
    print(f"MISSES {ds.name}: run {r['phase']}/{r['name']} ({setup.label()}, {r['setup'].get('embed')}), {k} chunks sent")
    print(f"  {len(missed)} of {len(rows)} questions miss at least one right passage.\n")
    count = {name: sum(1 for _, lab in missed if lab[0] == name) for name, _ in LABELS}
    print(f"  {'cause':<18}{'misses':>7}   go back to")
    for name, back in LABELS:
        print(f"  {name:<18}{count[name]:>7}   {back}")
    print()
    for row, (name, detail) in missed:
        q = ds.question(row["id"])
        print(f"  {row['n']:>3} {row['id']:<13} {name:<17} {detail}")
        print(f"      {q.text[:100]}")
    if missed:
        top = max(count, key=count.get)
        print(f"\n  Most common: {top} ({count[top]} of {len(missed)}). Look at one: python rag.py show "
              f"{next(row['n'] for row, lab in missed if lab[0] == top)} --run {r['phase']}/{r['name']}")
    return 0
