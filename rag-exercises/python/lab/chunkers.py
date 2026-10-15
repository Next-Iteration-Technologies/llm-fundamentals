"""Ways to cut documents into chunks. Every chunk keeps where it came from: file, start, end in the file's text.

A chunk's text may be changed on purpose (a sheet row with its column names written in, a mail without its quoted
replies), but its position never lies: all scoring uses positions (SPEC "One rule makes both datasets work").

A chunk is a dict: file, start, end, text, source (the label the model cites), and metadata where the dataset has it:
cls (data class), access, date, country, type, untrusted, parent (hierarchical: the section it hands over).
"""
import bisect, re

from . import datasets

TOKEN = re.compile(r"\w+|[^\w\s]")


def ntok(text):
    """Tokens, roughly: words and punctuation marks. Close enough to a model's count to compare sizes."""
    return len(TOKEN.findall(text))


# ------------------------------------------------------------------ labels and metadata
_headings = {}


def headings(ds, f):
    """(position, title) of every markdown heading in a file, to label chunks with the section they sit in."""
    k = (ds.name, f)
    if k not in _headings:
        _headings[k] = [(m.start(), m.group(1).strip()) for m in re.finditer(r"(?m)^#{1,6}\s+(.+)$", ds.docs[f])] \
            if f.endswith(".md") else []
    return _headings[k]


def label(ds, f, start):
    hs = headings(ds, f)
    i = bisect.bisect_right([p for p, _ in hs], start) - 1
    return f"{f} §{hs[i][1]}" if i >= 0 else f"{f} @{start}"


_file_meta = {}


def file_meta(ds):
    """Per-file metadata for chunkings that don't know a file's structure: the most restrictive data class of the
    file's structure-aware chunks (Contoso), else Public. Dates and country the same way."""
    if ds.name in _file_meta:
        return _file_meta[ds.name]
    meta = {f: {"cls": ds.cfg.get("class", "Public"), "access": ["team", "hr", "client"]} for f in ds.docs}
    if ds.cfg.get("chunking") == "structure":
        rank = {"Public": 0, "Internal": 1, "Confidential": 2}
        for c in chunk(ds, "structure"):
            m = meta[c["file"]]
            if rank.get(c.get("cls"), 1) >= rank.get(m.get("cls"), 0):
                m["cls"] = c.get("cls") or "Internal"
            m["date"] = max(m.get("date") or "", c.get("date") or "")
            m.setdefault("countries", set()).add(c.get("country"))
            m["access"] = sorted(set(m["access"]) & set(c.get("access", m["access"])) or set(c.get("access", [])))
        for m in meta.values():
            ps = m.pop("countries", set())
            m["country"] = ps.pop() if len(ps) == 1 else ""
    _file_meta[ds.name] = meta
    return meta


def make(ds, f, start, end, text=None, **extra):
    c = dict(file_meta(ds).get(f, {}), file=f, start=start, end=end, text=text if text is not None else ds.docs[f][start:end],
             source=label(ds, f, start))
    c.update(extra)
    return c


def finish(chunks):
    """Number the chunks: an id for each, stable for the same chunking of the same documents."""
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


# ------------------------------------------------------------------ fixed size
def fixed(ds, size=200, **_):
    """Every `size` tokens, no matter where that lands."""
    out = []
    for f, text in ds.docs.items():
        toks = [m.span() for m in TOKEN.finditer(text)]
        for i in range(0, len(toks), size):
            part = toks[i:i + size]
            out.append(make(ds, f, part[0][0], part[-1][1]))
    return finish(out)


# ------------------------------------------------------------------ structure-aware (Contoso, rag.py's own loaders)
def _sections(raw):
    """(heading, start, end) for every `## ` section of a markdown file, after the front matter."""
    fm = re.match(r"---\n(.*?)\n---\n", raw, re.S)
    body_at = fm.end() if fm else 0
    heads = list(re.finditer(r"(?m)^## +(.+)$", raw[body_at:]))
    return body_at, [(m.group(1).strip(), body_at + m.start(), body_at + (heads[n + 1].start() if n + 1 < len(heads) else len(raw) - body_at))
                     for n, m in enumerate(heads)]


def _line_spans(raw):
    pos, out = 0, []
    for line in raw.split("\n"):
        out.append((pos, pos + len(line), line))
        pos += len(line) + 1
    return out


def _record(raw, field, value):
    """The JSON object whose `field` is `value`."""
    m = re.search(rf'"{field}"\s*:\s*"{re.escape(value)}"', raw)
    return datasets.expand(raw, m.start(), m.end(), "object") if m else (0, len(raw))


def structure(ds, **_):
    """Contoso's structure-aware chunking, exactly as rag.py ingests it (E1): one policy or minutes section, one price
    list row, one mail message without quoted replies, one log line, one ticket per chunk. Positions are added here."""
    from . import host
    h = host()
    h.PACK = ds.cfg["_root"]
    chunks = h.load_pack()
    sections, lines, used = {}, {}, {}
    out = []
    for c in chunks:
        f, raw = c["file"], ds.docs[c["file"]]
        t = c["type"]
        if t in ("policy", "minutes", "onedrive"):
            if f not in sections:
                sections[f] = _sections(raw)
            body_at, secs = sections[f]
            if "section" not in c:  # a note without headings: the whole body
                s, e = body_at, len(raw)
            else:
                k = used.get(f, 0)
                while secs[k][0] != c["section"]:
                    k += 1
                used[f] = k + 1
                s, e = secs[k][1], secs[k][2]
        elif t == "mail":
            s, e = 0, len(raw)
        elif t == "price":
            ident = c["source"].split()[-1]
            m = re.search(rf"(?m)^{re.escape(ident)},.*$", raw)
            s, e = m.span()
        elif t == "log":
            if f not in lines:
                lines[f] = _line_spans(raw)
            k = used.get(f, 0)
            while re.sub(r"\s+", " ", lines[f][k][2]).strip() != c["text"]:
                k += 1
            used[f] = k + 1
            s, e = lines[f][k][0], lines[f][k][1]
        elif t == "ticket":
            s, e = _record(raw, "id", c["text"].split()[0])
        elif t == "jira":
            s, e = _record(raw, "key", c["source"].split()[-1])
        else:
            s, e = 0, len(raw)
        out.append(dict(c, start=s, end=e))
    return finish(out)


# ------------------------------------------------------------------ helpers for the text chunkings
def trim(text, s, e):
    """Shrink a span so it doesn't start or end with whitespace."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def cut_at(text, s, e, pattern):
    """Split the span [s, e) just before every match of pattern (a boundary), keeping every character."""
    points = [s] + [m.start() for m in re.compile(pattern).finditer(text, s, e) if s < m.start() < e] + [e]
    return [(a, b) for a, b in zip(points, points[1:]) if b > a]


HEADING = r"(?m)^#{1,6} "
PARAGRAPH = r"\n[ \t]*\n\s*"
LINE = r"\n"
SENTENCE = r"(?<=[.!?])\s+(?=\S)"


def sentence_spans(text):
    """Sentences, roughly: cut after . ! ? and at every line break (headings, table rows, list items are their own)."""
    out = []
    for a, b in cut_at(text, 0, len(text), LINE):
        for c, d in cut_at(text, a, b, SENTENCE):
            c, d = trim(text, c, d)
            if d > c:
                out.append((c, d))
    merged = []  # a piece of a few tokens ("Page 12", "---") joins the next one
    for c, d in out:
        if merged and ntok(text[merged[-1][0]:merged[-1][1]]) < 4:
            merged[-1] = (merged[-1][0], d)
        else:
            merged.append((c, d))
    return merged


def token_windows(text, s, e, size):
    toks = [m.span() for m in TOKEN.finditer(text, s, e)]
    return [(toks[i][0], toks[min(i + size, len(toks)) - 1][1]) for i in range(0, len(toks), size)]


def recursive_spans(text, s, e, size, levels=(HEADING, PARAGRAPH, LINE, SENTENCE)):
    """Headings first, then paragraphs, then lines, then sentences, then (last resort) fixed windows; small neighbours
    are merged back up to `size` tokens, but never across a heading."""
    if ntok(text[s:e]) <= size:
        return [trim(text, s, e)]
    if not levels:
        return token_windows(text, s, e, size)
    parts = cut_at(text, s, e, levels[0])
    if len(parts) == 1:
        return recursive_spans(text, s, e, size, levels[1:])
    pieces = [sp for a, b in parts for sp in recursive_spans(text, a, b, size, levels[1:])]
    if levels[0] == HEADING:
        return pieces
    out = []
    for a, b in pieces:
        if out and ntok(text[out[-1][0]:b]) <= size:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _spans_to_chunks(ds, f, spans):
    text = ds.docs[f]
    out = []
    for a, b in spans:
        a, b = trim(text, a, b)
        if b > a and TOKEN.search(text, a, b):
            out.append(make(ds, f, a, b))
    return out


# ------------------------------------------------------------------ the other chunkings
def naive(ds, size=500, **_):
    """E1's naive baseline: every `size` characters, no structure at all."""
    return finish([make(ds, f, i, min(i + size, len(t))) for f, t in ds.docs.items() for i in range(0, len(t), size)
                   if t[i:i + size].strip()])


def recursive(ds, size=300, **_):
    """Headings first, then paragraphs, then sentences, so pieces end at natural breaks. Up to `size` tokens."""
    return finish([c for f, t in ds.docs.items() for c in _spans_to_chunks(ds, f, recursive_spans(t, 0, len(t), size))])


def sentence(ds, size=4, **_):
    """A few sentences per chunk (`size` sentences)."""
    out = []
    for f, t in ds.docs.items():
        sp = sentence_spans(t)
        out += _spans_to_chunks(ds, f, [(sp[i][0], sp[min(i + size, len(sp)) - 1][1]) for i in range(0, len(sp), size)])
    return finish(out)


def boundary_model(ds):
    """The model semantic chunking uses to see topic changes: one fixed, fast model (the dataset's `semantic_embed`,
    else bge-small if providers.toml has it, else hash), so the chunks don't change when the embedding phase changes the
    search model, and nobody waits hours for a slow model to embed every sentence."""
    import models
    name = ds.cfg.get("semantic_embed") or ("bge-small" if "bge-small" in models.PROVIDERS else "hash")
    return name


def semantic(ds, size=300, **_):
    """A new chunk where the topic changes: neighbouring sentences whose embeddings are least alike (the lowest
    quarter of similarities in the file) start a new chunk. Never more than `size` tokens."""
    import sys
    import numpy as np
    import models
    from . import pipeline
    out = []
    per_file = {f: sentence_spans(t) for f, t in ds.docs.items()}
    texts = [ds.docs[f][a:b] for f, sp in per_file.items() for a, b in sp]
    model = boundary_model(ds)
    try:
        vecs, _ = pipeline.embed(texts, "doc", model, progress=f"{ds.name} semantic chunking: sentences")
    except models.SetupProblem as e:
        print(f"! semantic chunking: {model} unavailable ({e}); seeing topic changes with hash instead.", file=sys.stderr)
        vecs, _ = pipeline.embed(texts, "doc", "hash")
    k = 0
    for f, sp in per_file.items():
        v = vecs[k:k + len(sp)]
        k += len(sp)
        if not sp:
            continue
        sims = np.sum(v[1:] * v[:-1], axis=1) if len(sp) > 1 else np.array([])
        cut = np.percentile(sims, 25) if len(sims) else 0
        spans, cur, t = [], [sp[0][0], sp[0][1]], ds.docs[f]
        for i in range(1, len(sp)):
            n = ntok(t[cur[0]:sp[i][1]])
            if sims[i - 1] < cut or n > size:
                spans.append(tuple(cur))
                cur = [sp[i][0], sp[i][1]]
            else:
                cur[1] = sp[i][1]
        spans.append(tuple(cur))
        out += _spans_to_chunks(ds, f, spans)
    return finish(out)


def hierarchical(ds, size=800, child=2, **_):
    """Search small pieces, send their whole section: parents are sections (cut at headings, at most `size` tokens),
    children are `child` sentences each. The children are searched; a hit hands over its parent."""
    parents, children = [], []
    for f, t in ds.docs.items():
        for a, b in recursive_spans(t, 0, len(t), size, levels=(HEADING, PARAGRAPH, LINE, SENTENCE)):
            a, b = trim(t, a, b)
            if b <= a or not TOKEN.search(t, a, b):
                continue
            pid = len(parents)
            parents.append(make(ds, f, a, b, id=pid))
            sp = [x for x in sentence_spans(t[a:b])]
            for i in range(0, len(sp), child):
                s0, e0 = a + sp[i][0], a + sp[min(i + child, len(sp)) - 1][1]
                children.append(make(ds, f, s0, e0, parent=pid))
    return parents, children


CONTEXT_PROMPT = ("Document: {doc}\nChunk: {chunk}\nWrite one sentence that says where this chunk sits in the document "
                  "and what it is about. Only the sentence.")


def contextual(ds, size=300, chat=None, limit=None, base="recursive", embed=None, **_):
    """Any chunking plus one sentence, written by the chat model, saying where the chunk comes from. The sentence is
    searched with the chunk (embed_text); the model is sent the chunk itself. Only for chunks the model may read.
    limit: write sentences for the first N chunks only (a quick test); the rest are searched as they are."""
    from . import llm, pipeline
    import models
    base_chunks = chunk(ds, base if size == 300 else f"{base}-{size}")
    out, may = [], models.MAY_READ[models.where(chat)]
    system = "You write short context prefixes for search."
    for n, c in enumerate(base_chunks):
        c = dict(c)
        if (limit is None or n < limit) and c.get("cls", "Public") in may and c.get("type") != "log":
            doc = ds.docs[c["file"]]
            head_ = doc[:1500]
            r = llm.ask(chat, system, CONTEXT_PROMPT.format(doc=f"{c['file']}\n{head_}", chunk=c["text"]))
            c["context"] = r["text"].strip()
            c["embed_text"] = f"{c['context']} {c['text']}"
        out.append(c)
    return finish(out)


CHUNKERS = {"fixed": fixed, "structure": structure, "naive": naive, "recursive": recursive, "sentence": sentence,
            "semantic": semantic, "hierarchical": hierarchical, "contextual": contextual}
NEEDS_EMBED = set()   # chunkings whose chunks depend on the run's embed model (none: semantic uses boundary_model)
NEEDS_CHAT = {"contextual"}


def parse(name):
    """'fixed-200' -> (fixed, {size: 200}); 'structure' -> (structure, {})."""
    m = re.match(r"^([a-z]+)(?:-(\d+))?$", name)
    if not m or m.group(1) not in CHUNKERS:
        raise SystemExit(f"! no chunking '{name}'. Have: " + ", ".join(sorted(CHUNKERS)) + " (some take a size: fixed-200).")
    return CHUNKERS[m.group(1)], ({"size": int(m.group(2))} if m.group(2) else {})


_cache, _units = {}, {}


def cache_key(ds, name, embed=None, chat=None, limit=None):
    kind = parse(name)[0].__name__
    return (ds.name, name, embed if kind in NEEDS_EMBED else None, (chat, limit) if kind in NEEDS_CHAT else None)


def chunk(ds, name, embed=None, chat=None, limit=None):
    """The chunks of a dataset for one chunking (made once per run). Semantic chunking depends on the embed model,
    contextual on the chat model."""
    k = cache_key(ds, name, embed, chat, limit)
    if k not in _cache:
        fn, kw = parse(name)
        out = fn(ds, embed=embed, chat=chat, limit=limit, **kw)
        if isinstance(out, tuple):  # hierarchical: (parents, children)
            out, _units[k] = out
        _cache[k] = out
    return _cache[k]


def units(ds, name, embed=None, chat=None, limit=None):
    """What search looks at: the chunks themselves, or for hierarchical chunking the small children
    (each child carries `parent`, the id of the chunk it hands over)."""
    chunk(ds, name, embed, chat, limit)
    k = cache_key(ds, name, embed, chat, limit)
    return _units.get(k) or _cache[k]
