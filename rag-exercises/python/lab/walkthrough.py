"""python rag.py walkthrough: one question through the whole pipeline, each step printed in plain words (phase 0)."""
import statistics, sys, textwrap, time

import models

from . import chunkers, datasets, llm, metrics, pipeline

W = 110


def short(text, n=100):
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n - 1] + "…"


def head(n, name, what):
    print(f"\n{n}. {name.upper()}: {what}")


def show_hits(hits, n=10, gold=()):
    for r, (c, s) in enumerate(hits[:n], 1):
        mark = " *" if any(metrics.matches(c, g) for g in gold) else "  "
        print(f"   {r:>2}.{mark}{s:>8.4f}  [{c['source'][:48]}]  {short(c['text'], 60)}")


def run(a):
    ds = datasets.load(a.dataset)
    q = ds.question(a.question_n) if a.question_n else None
    if not q and not a.question:
        sys.exit("! give a question in quotes, or --question N to take question N from the golden set.")
    text = q.text if q else a.question
    setup = pipeline.Setup(chunking=a.chunking or ds.chunking, retriever=a.retriever, rerank=a.rerank, embed=a.embed or "", k=a.k)
    if getattr(a, "compare_with", None):
        from . import results
        return results.walkthrough_compare(a, ds, q, text, setup)
    walk(a, ds, q, text, setup)


def walk(a, ds, q, text, setup, title=""):
    t_all = time.time()
    chat = None if a.dry else llm.chat_name(a.chat)
    print(f"WALKTHROUGH{title}  dataset {ds.name} ({ds.title})")
    print(f"  question{f' {q.n} ({q.id})' if q else ' (typed in)'}: {text}")
    print(f"  setup: chunking {setup.chunking}, search {setup.retriever}{' + rerank' if setup.rerank else ''}, "
          f"{setup.k} chunks sent" + (f", chat {llm.describe(chat)}" if chat else ", no chat model (--dry)"))

    # 1. load
    chars = sum(len(t) for t in ds.docs.values())
    head(1, "Load", f"read {len(ds.docs)} documents, {chars:,} characters, from {ds.cfg['docs']}")
    files = list(ds.docs)
    print("   " + ", ".join(files[:4]) + (f", … and {len(files) - 4} more" if len(files) > 4 else ""))

    # 2. chunk
    index = pipeline.build(ds, setup, a.dry)
    chunks = index.chunks
    sizes = [chunkers.ntok(c["text"]) for c in chunks]
    head(2, "Chunk", f"cut them into {len(chunks):,} chunks with '{setup.chunking}'; "
                     f"{min(sizes)} to {max(sizes)} tokens each (median {int(statistics.median(sizes))})")
    for c in chunks[:: max(1, len(chunks) // 3)][:3]:
        print(f"   [{c['source'][:50]}] chars {c['start']}-{c['end']}: {short(c['text'], 70)}")

    # 3. embed
    units = index.units
    vec_len = index.vecs.shape[1]
    head(3, "Embed", f"{pipeline.embed_desc(index.embed)} turned each of the {len(units):,} "
                     f"{'search pieces' if len(units) != len(chunks) else 'chunks'} into {vec_len} numbers "
                     f"({index.new} new, {len(units) - index.new} saved from an earlier run), and the question too")

    # 4. search
    final, first = pipeline.retrieve(index, text, setup)
    gold = q.gold if q else []
    head(4, "Search", f"{setup.retriever} search: the {len(first)} best chunks for the question; the top 10"
                      + (" (* = a gold passage)" if gold else ""))
    show_hits(first, 10, gold)

    # 5. rerank
    if setup.rerank:
        head(5, "Rerank", f"a second, slower model ({pipeline.RERANKER}) reread the top {len(first)} with the question; new order:")
        show_hits(final, 10, gold)
    elif a.no_rerank_demo:
        head(5, "Rerank", "skipped (--no-rerank).")
    else:
        try:
            demo = pipeline.rerank(text, first)
            head(5, "Rerank", f"not part of this setup (Chapter 4 tries it), so the prompt keeps the search order. "
                              f"For a look, the order a second, slower model ({pipeline.RERANKER}) would give:")
            show_hits(demo, 10, gold)
        except Exception as e:
            head(5, "Rerank", f"skipped: the reranker is not available ({type(e).__name__}: {str(e)[:100]}).")
    if a.dry:
        print("\n--dry: stopping after step 5 (rerank). Steps 6 to 8 (prompt, answer, check) run without --dry.")
        return None

    # 6. prompt
    sent, held, where = pipeline.residency([c for c, _ in final[:setup.k]], chat)
    system = pipeline.system_prompt(ds)
    prompt = pipeline.build_prompt(text, sent)
    ptok = chunkers.ntok(system) + chunkers.ntok(prompt)
    head(6, "Prompt", f"exactly what the model is sent: about {ptok:,} tokens (instructions {chunkers.ntok(system)}, "
                      f"{len(sent)} chunks and the question {chunkers.ntok(prompt)})")
    if held:
        print(f"   held back from a '{where}' model (data class): " + ", ".join(f"[{c['source']}] {c.get('cls')}" for c in held))
    print("   ---- system ----")
    print(textwrap.indent(system, "   "))
    print("   ---- message ----")
    print(textwrap.indent(prompt, "   "))
    print("   ----")

    # 7. answer
    try:
        r = llm.ask(chat, system, prompt, fresh=a.fresh)
    except models.SetupProblem as e:
        print(f"\n! {e}")
        sys.exit(2)
    when = f"saved reply from {r['saved']}" if r["saved"] else f"{r['seconds']} s"
    head(7, "Answer", f"{llm.describe(chat)} answered ({when}):")
    print(textwrap.indent(textwrap.fill(r["text"], W, replace_whitespace=False) if "\n" not in r["text"] else r["text"], "   "))
    import re
    cited = sorted(set(re.findall(r"\[([^\]]+)\]", r["text"])))
    print(f"   cited: {', '.join(cited) if cited else 'nothing'}")

    # 8. check
    if not q:
        head(8, "Check", "skipped: a typed question has no known answer. Use --question N to pick one from the golden set.")
    elif not gold:
        head(8, "Check", "this question has no gold passages" + (": the right answer is to say it isn't in the documents."
                                                                  if q.refuse else "."))
    else:
        ranked = [c for c, _ in final]
        ranks = metrics.gold_ranks(ranked, gold)
        head(8, "Check", f"the {len(gold)} gold passage{'s' if len(gold) > 1 else ''} for this question, and whether we found them")
        for g, r in zip(gold, ranks):
            state = f"found (rank {r})" + (", sent" if r <= setup.k else ", not sent") if r else "not found"
            print(f"   {state:<24} [{g.file} {g.start}-{g.end}] {short(ds.docs[g.file][g.start:g.end], 60)}")
        s = metrics.score(ranks, [g.grade for g in gold])
        print(f"   {sum(1 for r in ranks if r and r <= setup.k)} of {len(gold)} in the {setup.k} chunks sent, "
              f"{sum(1 for r in ranks if r)} in the top {len(ranked)}.  Recall@3 {s['R@3']:.2f}, MRR {s['MRR']:.2f}")
    print(f"\n({time.time() - t_all:.0f} s)")
    return final
