"""The lab's commands, added to rag.py's parser. See LAB.md for the story and .plan/SPEC.md for the design.

Every lab command takes --dataset NAME (datasets/NAME.toml). Without it: RAG_DATASET, and without that: contoso.
"""
import os, sys


def dataset_arg(p):
    p.add_argument("--dataset", metavar="NAME", help="datasets/NAME.toml (default: RAG_DATASET, else contoso)")


def setup_args(p, chunking=True):
    if chunking:
        p.add_argument("--chunking", metavar="NAME", help="how to cut the documents (default: the dataset's)")
    p.add_argument("--retriever", choices=["keyword", "semantic", "hybrid"], default="semantic")
    p.add_argument("--rerank", action="store_true", help="rerank the top 50 with a cross-encoder")
    p.add_argument("--embed", metavar="NAME", help="embed block (default: LLM_EMBED, else the dataset's, else providers.toml)")
    p.add_argument("--k", type=int, default=5, help="chunks sent to the model")


def eval_args(p):
    """Added to rag.py's own eval. Any of them, or RAG_DATASET, switches eval to the lab's version."""
    dataset_arg(p)
    p.add_argument("--retriever", choices=["keyword", "semantic", "hybrid", "perfect", "random"],
                   help="lab: how to search; perfect and random check the test itself")
    p.add_argument("--chunking", metavar="NAME", help="lab: how to cut the documents (default: the dataset's)")
    p.add_argument("--rerank", action="store_true", help="lab: rerank the top 50")
    p.add_argument("--save", metavar="NAME", help="lab: save the run as eval/NAME (default: baseline for the starting setup)")


def lab_eval(a):
    return a.cmd == "eval" and bool(a.dataset or os.environ.get("RAG_DATASET") or a.retriever or a.save or a.chunking or a.rerank)


def add_commands(sub):
    p = sub.add_parser("selftest", help="the build checks: PASS/FAIL per check")
    p.add_argument("--upto", type=int, default=99, help="run the checks of steps 0..N")
    p.add_argument("--only", metavar="TEXT", help="only checks whose name contains TEXT")
    p.add_argument("--save-baseline", action="store_true", help="record today's Contoso numbers in .plan/baseline.json")

    p = sub.add_parser("check-data", help="is every gold passage where the golden set says?")
    dataset_arg(p)

    p = sub.add_parser("walkthrough", help="one question through the whole pipeline, step by step")
    dataset_arg(p)
    p.add_argument("question", nargs="?", help="a question of your own (step 8 is skipped: nobody marked its answer)")
    p.add_argument("--question", dest="question_n", metavar="N", help="question N (or its id) from the golden set")
    setup_args(p)
    p.add_argument("--no-rerank", dest="no_rerank_demo", action="store_true", help="don't show what the reranker would do")
    p.add_argument("--chat", metavar="NAME", help="chat block that answers (default: LLM_CHAT, else providers.toml)")
    p.add_argument("--dry", action="store_true", help="no model: stop after step 5")
    p.add_argument("--fresh", action="store_true", help="ask the model again instead of reusing a saved reply")
    p.add_argument("--compare-with", metavar="RUN", help="also run it with RUN's setup, side by side")


    p = sub.add_parser("chunks", help="what each chunking looks like: count, sizes, gold passages cut in two")
    dataset_arg(p)
    p.add_argument("--chunking", metavar="NAME", help="only this chunking")
    p.add_argument("--embed", metavar="NAME", help="embed block (semantic chunking uses it to see topic changes)")
    p.add_argument("--chat", metavar="NAME", help="chat block for contextual chunking")
    p.add_argument("--limit", type=int, help="contextual: write context sentences for the first N chunks only")
    p.add_argument("--show", type=int, default=0, metavar="N", help="print N example chunks of each")

    p = sub.add_parser("compare", help="score every setup of one phase (experiments/PHASE.toml)")
    dataset_arg(p)
    p.add_argument("--phase", required=True, help="chunking, retrieval, embedding or filters")
    p.add_argument("--dry", action="store_true", help="no chat model: rows that need one are skipped")
    p.add_argument("--embed", metavar="NAME", help="embed block (default: LLM_EMBED, else the dataset's, else providers.toml)")
    p.add_argument("--chat", metavar="NAME", help="chat block for rows that need a model (rewrites, contextual)")
    p.add_argument("--rows", metavar="A,B", help="only these rows")
    p.add_argument("--skip", metavar="A,B", help="leave these rows out")
    p.add_argument("--limit", type=int, help="contextual: write context sentences for the first N chunks only")
    p.add_argument("--chunkings", metavar="A,B", help="retrieval: these chunkings instead of the two best and fixed-200")
    p.add_argument("--setup", metavar="RUN", help="embedding, filters: keep this saved run's setup instead of the best so far")

    p = sub.add_parser("show", help="one question: the gold passages next to what the run found")
    dataset_arg(p)
    p.add_argument("question", help="question number (or id) in the golden set")
    p.add_argument("--run", metavar="RUN", help="phase/name of a saved run (default: eval/baseline)")
    p.add_argument("--best", action="store_true", help="the best run so far")

    p = sub.add_parser("answers", help="answer the golden questions with a run's setup and score the answers (RAG triad)")
    dataset_arg(p)
    p.add_argument("run", nargs="?", help="phase/name of a saved run (default: the best so far)")
    p.add_argument("--best", action="store_true", help="the best run so far (by Recall@5, then MRR)")
    p.add_argument("--limit", type=int, help="only N questions, spread over the golden set")
    p.add_argument("--chat", metavar="NAME", help="chat block that answers (default: LLM_CHAT, else providers.toml)")
    p.add_argument("--judge", metavar="NAME", help="chat block that judges (default: LLM_JUDGE, else `judge` in providers.toml, else --chat)")

    p = sub.add_parser("misses", help="why does it still miss? one label per missed question")
    dataset_arg(p)
    p.add_argument("run", nargs="?", help="phase/name of a saved run (default: the best so far)")
    p.add_argument("--best", action="store_true", help="the best run so far")

    p = sub.add_parser("final", help="RAG vs the whole corpus in the prompt, same model (LAB chapter 9)")
    dataset_arg(p)
    p.add_argument("--model", metavar="NAME", help="chat block for both rows (default: LLM_CHAT, else providers.toml)")
    p.add_argument("--judge", metavar="NAME", help="chat block that judges both rows")
    p.add_argument("--setup", metavar="RUN", help="the RAG setup: phase/name of a saved run (default: the best so far)")
    p.add_argument("--limit", type=int, help="only N questions, spread over the golden set")
    p.add_argument("--yes", action="store_true", help="send without asking (the token preview is still printed)")
    p.add_argument("--ignore-data-class", action="store_true", help="Contoso: let the model read every class (fictional pack only)")

    p = sub.add_parser("report", help="rebuild python/results/RESULTS.md from the saved runs")
    dataset_arg(p)


COMMANDS = {"selftest", "check-data", "walkthrough", "report", "chunks", "compare", "show", "answers", "misses", "final"}


def run(a):
    if a.cmd == "eval":
        return run_eval(a)
    if a.cmd == "selftest":
        from . import selftest
        if a.save_baseline:
            selftest.save_baseline()
            return 0
        return selftest.run(a.upto, a.only)
    if a.cmd == "check-data":
        from . import datasets
        return 1 if datasets.check_data(datasets.load(a.dataset)) else 0
    if a.cmd == "walkthrough":
        from . import walkthrough
        walkthrough.run(a)
        return 0
    if a.cmd in ("chunks", "compare", "show"):
        from . import compare
        if a.cmd == "show" and a.best:
            a.run = "--best"
        return {"chunks": compare.chunks_cmd, "compare": compare.compare, "show": compare.show_cmd}[a.cmd](a) or 0
    if a.cmd == "final":
        from . import final
        return final.run(a)
    if a.cmd == "misses":
        from . import misses
        return misses.run(a)
    if a.cmd == "answers":
        from . import answers
        return answers.run(a)
    if a.cmd == "report":
        from . import results
        out = results.report()
        text = out.read_text(encoding="utf-8")
        ds = a.dataset or os.environ.get("RAG_DATASET")
        if ds:  # only the chosen dataset's part
            import re
            m = re.search(rf"^## {re.escape(ds)}\n.*?(?=^## |\Z)", text, re.S | re.M)
            text = m.group(0) if m else f"(no runs for {ds} yet)\n"
        print(text)
        print(f"written: {out}")
        return 0


def run_eval(a):
    """LAB chapter 2: every golden question through one setup, scored and saved; perfect and random check the test."""
    from . import datasets, llm, pipeline, results
    ds = datasets.load(a.dataset)
    kind = a.retriever if a.retriever in ("perfect", "random") else "pipeline"
    setup = pipeline.Setup(chunking=a.chunking or ds.chunking,
                           retriever=a.retriever if a.retriever in ("keyword", "semantic", "hybrid") else "semantic",
                           rerank=a.rerank, embed=a.embed or "", k=a.k, transform=(a.transform or "") if not a.dry else "")
    default = pipeline.Setup(chunking=ds.chunking, k=a.k, embed=setup.embed)  # any embedding: still the starting setup
    name = a.save or (kind if kind != "pipeline" else "baseline" if setup == default else
                      f"{setup.chunking}-{setup.retriever}{'-rerank' if setup.rerank else ''}{'-' + setup.transform if setup.transform else ''}")
    chat = llm.chat_name(a.chat) if setup.transform else None
    r = results.run(ds, setup, "eval", name, retriever=kind, dry=a.dry, chat=chat)
    results.print_run(r, ds)
    p = results.save(r)
    results.report()
    print(f"\n  saved as eval/{name}: {p}")
    return 0
