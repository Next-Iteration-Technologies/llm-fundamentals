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


def is_held(c, held):
    return any(c is h for h in held)


def say(text):
    print(textwrap.fill(text, W, initial_indent="   ", subsequent_indent="   "))


def explain(ds, setup, final, demo, sent, held, where, cited, gold, ranks):
    """Step 9 (--explain): the run in plain words, from what steps 4 to 8 already know. No model is asked."""
    head(9, "Explain", "what happened, in plain words")
    k, n = min(setup.k, len(final)), len(final)

    # what was sent, and in which order
    say(f"The prompt took the top {k} in the reranker's order (step 5)." if setup.rerank else
        f"The prompt took the top {k} in the search order (step 4). Rerank is not part of this setup"
        + (": the step 5 list was only a preview and changed nothing." if demo else "."))
    for r, (c, _) in enumerate(final[:k], 1):
        mark = "*" if any(metrics.matches(c, g) for g in gold) else " "
        print(f"   {r:>2}. {mark} {'held back' if is_held(c, held) else 'sent':<9}  [{c['source'][:70]}]")

    # the data-class rule (flow 9.4)
    if held:
        may = ", ".join(sorted(x for x in models.MAY_READ[where] if x))
        classes = ", ".join(sorted({c.get("cls") or "no class" for c in held}))
        if sent:
            say(f"{len(held)} of the {k} were held back: a '{where}' chat model may read {may}, and they are {classes}. "
                f"The model answered from the other {len(sent)}.")
        else:
            say(f"All {k} were held back: a '{where}' chat model may read {may}, and they are {classes}. "
                f"The model got no passages at all, so an \"I don't find this\" comes from the data-class rule, not from search.")
        say("To let the model read them: --chat NAME with a block whose where is \"tenant\" or \"local\", "
            "CHAT_MODEL_WHERE=tenant for one run, or where = \"tenant\" under the block in providers.toml.")
    if not ranks:
        return

    # where each gold passage was
    order = "reranked" if setup.rerank else "search"
    preview = metrics.gold_ranks([c for c, _ in demo], gold) if demo else [None] * len(gold)
    print()
    say(f"Where each right passage (gold, * above) was, in the {order} order:")
    for g, r, dr in zip(gold, ranks, preview):
        label = f"[{g.file}] \"{short(ds.docs[g.file][g.start:g.end], 50)}\""
        if r and r <= k:
            why = f"rank {r}: " + ("in the top k, but held back (data class)." if is_held(final[r - 1][0], held) else "sent.")
        elif r:
            why = f"rank {r}: in the top {n}, but not in the {k} sent."
            if dr:
                why += f" The reranker would move it to {dr}, " + (f"inside the {k}." if dr <= k else f"still outside the {k}.")
        else:
            why = f"not in the top {n} at all."
            same = next(((i, c) for i, (c, _) in enumerate(final, 1) if c["file"] == g.file), None)
            if same:
                why += (f" A chunk from the same file is at rank {same[0]} ([{same[1]['source'][:50]}]), but it's another part "
                        f"of the file: gold counts by character range, so it doesn't match.")
        say(f"- {label}: {why}")

    # the scores, worked out
    s = metrics.score(ranks, [g.grade for g in gold])
    hit = [r for r in ranks if r]
    print()
    say(f"The scores (they use the {order} order" + (", not the step 5 preview):" if demo else "):"))
    for key, cut in (("R@3", 3), ("R@5", 5), ("R@50", 50)):
        got, out_of = sum(1 for r in ranks if r and r <= cut), min(cut, len(ranks))
        say(f"- Recall@{cut} = {got} of {out_of} = {s[key]:.2f}. {metrics.EXPLAIN[key]}")
    say(f"- MRR = 1 / {min(hit)} = {s['MRR']:.2f}, because the first right chunk is at rank {min(hit)}." if hit else
        f"- MRR = 0, because no right chunk is in the top {n}.")
    say(f"  {metrics.EXPLAIN['MRR']}")
    if held:
        say("  The scores measure search only: a held-back chunk still counts as found.")

    # the answer next to retrieval
    if sent:
        by_src = {c["source"]: c for c in sent}
        cg = [x for x in cited if x in by_src and any(metrics.matches(by_src[x], g) for g in gold)]
        other = [x for x in cited if x in by_src and x not in cg]
        unknown = [x for x in cited if x not in by_src]
        print()
        if not cited:
            say("The answer cites nothing.")
        else:
            say(f"The answer cites {len(cited)} passage{'s' if len(cited) > 1 else ''}: {len(cg)} gold, {len(other)} not gold."
                + (f" {len(unknown)} citation{'s' if len(unknown) > 1 else ''} match no passage that was sent: check them."
                   if unknown else ""))
        if other or (cited and not cg):
            say("A chunk that isn't gold can still hold the answer, for example the same fact in another note. "
                "The scores above judge the search; Chapter 7 scores the answers themselves.")

    # what to try
    tips = []
    if held and (not sent or any(r and r <= k and is_held(final[r - 1][0], held) for r in ranks)):
        tips.append("let the model read the held-back passages (see above)")
    if any(r and r > k for r in ranks):
        tips.append(f"gold in the top {n} but past the {k} sent: try --rerank, --retriever hybrid, or a larger --k")
    if any(r is None for r in ranks):
        tips.append(f"gold not in the top {n}: try --retriever hybrid, or another --chunking "
                    f"(python rag.py chunks shows which chunkings cut a gold passage in two)")
    print()
    say("What to try: " + "; ".join(tips) + "." if tips else
        f"Every right passage was in the {k} sent: search did its job for this question.")


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
    print(f"  setup: chunking {setup.chunking}, search {setup.retriever}{' + rerank' if setup.rerank else ''}"
          + "".join(f", {k}={v}" for k, v in sorted(setup.filters.items()))
          + f", embedding {pipeline.embed_name(ds, setup.embed or None)}, {setup.k} chunks sent"
          + (f", chat {llm.describe(chat)}" if chat else ", no chat model (--dry)"))

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
    demo = None
    if setup.rerank:
        head(5, "Rerank", f"a second, slower model ({models.rerank_id()}) reread the top {len(first)} with the question; new order:")
        show_hits(final, 10, gold)
    elif a.no_rerank_demo:
        head(5, "Rerank", "skipped (--no-rerank).")
    else:
        try:
            demo = pipeline.rerank(text, first)
            head(5, "Rerank", f"not part of this setup (Chapter 4 tries it), so the prompt keeps the search order. "
                              f"For a look, the order a second, slower model ({models.rerank_id()}) would give:")
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
    ranks = None
    if q and gold:
        ranked = [c for c, _ in final]
        ranks = metrics.gold_ranks(ranked, gold)
        head(8, "Check", f"the {len(gold)} gold passage{'s' if len(gold) > 1 else ''} for this question, and whether we found them")
        for g, r in zip(gold, ranks):
            state = f"found (rank {r})" + (", not sent" if r > setup.k else ", held back" if is_held(ranked[r - 1], held)
                                           else ", sent") if r else "not found"
            print(f"   {state:<26} [{g.file} {g.start}-{g.end}] {short(ds.docs[g.file][g.start:g.end], 60)}")
        s = metrics.score(ranks, [g.grade for g in gold])
        n_sent = sum(1 for r in ranks if r and r <= setup.k and not is_held(ranked[r - 1], held))
        print(f"   {n_sent} of {len(gold)} in the {len(sent)} chunks sent, "
              f"{sum(1 for r in ranks if r)} in the top {len(ranked)}.  Recall@3 {s['R@3']:.2f}, MRR {s['MRR']:.2f}")
    if getattr(a, "explain", False):
        explain(ds, setup, final, demo, sent, held, where, cited, gold, ranks)
    print(f"\n({time.time() - t_all:.0f} s)")
    return final
