"""Retrieval as settings: keyword, semantic, hybrid (RRF), + rerank (lab/pipeline.py), query rewriting (here),
and the search filters of phase 5 (here). Query rewriting follows rag.py's E8 (flow 4.6), with the replies saved."""
import re, sys

import models

from . import host, llm

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


COUNTRIES = {"UK": ["uk", "united kingdom", "britain", "london"], "Netherlands": ["netherlands", "dutch", "holland", "utrecht"],
             "Germany": ["germany", "german", "cologne"]}


def countries_in(text):
    """The countries a question names, by name, adjective or office city."""
    return [c for c, words in COUNTRIES.items() if any(re.search(rf"\b{w}\b", text.lower()) for w in words)]


def passes(chunk, filters, ds, q=None):
    """Phase 5 filters, applied before ranking (like rag.py's permission filter):
      asker = "golden"  each question asked as its golden-set asker: only chunks that role may see (E6)
      country = "question"  a question that names a country only searches that country (and "All")
      after = "YYYY-MM-DD"  only documents dated on or after this day (undated ones stay)"""
    if filters.get("asker"):
        role = (q.asker if q and filters["asker"] == "golden" else filters["asker"]) or "employee"
        if not set(chunk.get("access", ["team", "hr", "client"])) & set(host().ROLES.get(role, [role])):
            return False
    if filters.get("country") and q is not None:
        named = countries_in(q.text)
        if named and chunk.get("country", "") not in named + ["", "All"]:
            return False
    if filters.get("after") and chunk.get("date") and str(chunk["date"]) < filters["after"]:
        return False
    return True


def prefer_newest(hits, k):
    """When sources disagree, prefer the newest: the k chunks that are sent stay the same, but newest first, so a newer
    source sits above an older one it contradicts (and the model reads it first). Undated chunks keep their place
    after the dated ones. A bonus for being new on every candidate pulled in recent but unrelated mail and logs."""
    top, rest = hits[:k], hits[k:]
    dated = sorted([h for h in top if h[0].get("date")], key=lambda h: str(h[0]["date"]), reverse=True)
    return dated + [h for h in top if not h[0].get("date")] + rest
