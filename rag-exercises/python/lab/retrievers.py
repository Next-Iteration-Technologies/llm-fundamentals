"""Retrieval as settings: keyword, semantic, hybrid (RRF), + rerank (lab/pipeline.py), query rewriting (here),
and the search filters of phase 5 (here). Query rewriting follows rag.py's E8 (flow 4.6), with the replies saved."""
import re, sys

import models

from . import llm

PLAIN = " It goes to a search over the documents, not the web: plain words, no site:, quotes or operators."


def transform(question, how, chat, dry=False):
    """The searches actually run, as [(keyword text, vector text, embed kind)]. rewrite: one better query;
    multi: the question plus three versions; hyde: search with a drafted answer (it only has to sound right)."""
    if dry or not how:
        if how:
            print(f"! --dry: '{how}' needs the chat model; searching the question as asked.", file=sys.stderr)
        return [(question, question, "query")]
    try:
        return _transform(question, how, chat)
    except models.SetupProblem as e:
        sys.exit(f"! query transformation '{how}' needs the chat model: {e}")


def _transform(question, how, chat):
    system = "You write search queries."
    if how == "rewrite":
        r = llm.ask(chat, system, f"Question: {question}\nRewrite it as one search query in the words the documents "
                                  "would use. Only the query." + PLAIN)["text"]
        return [(r, r, "query")]
    if how == "multi":
        out = llm.ask(chat, system, f"Question: {question}\nWrite three different search queries that together cover every "
                                    "part of it, in the words the documents would use. One per line, nothing else." + PLAIN)["text"]
        lines = [re.sub(r"^[\s\d.\-*)]+", "", l).strip('" ') for l in out.splitlines()]
        return [(question, question, "query")] + [(l, l, "query") for l in lines if l][:3]
    if how == "hyde":
        d = llm.ask(chat, "You write example documents.",
                    f"Write a short passage, as it would appear in the documents, that answers: {question}\n"
                    "Invent plausible details; it is only used for search.")["text"]
        return [(question, d, "doc")]
    sys.exit(f"! no query transformation '{how}': rewrite, multi or hyde.")


def passes(chunk, filters, ds):
    """Does a chunk pass the filters? Filled in by phase 5; no filters: everything passes."""
    return True
