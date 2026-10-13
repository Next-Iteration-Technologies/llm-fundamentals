"""Any dataset from datasets/<name>.toml: its documents, its questions, and every kind of gold turned into file positions.

Gold comes in three kinds, all ending as (file, start, end) in the file's text:
  position  the golden set gives the character positions (D&D)
  quote     the golden set quotes the text; it is found in the file (Contoso, through corpus-index.json lab_match)
  file      only the file is known: the whole file is the passage
A question without gold is a refusal question: the answer must say it doesn't find it.
"""
import json, os, re, sys
from dataclasses import dataclass, field
from pathlib import Path

from . import ROOT

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

DATASETS = ROOT / "datasets"
DEFAULT = "contoso"


@dataclass
class Gold:
    file: str
    start: int
    end: int
    grade: int = 1          # 3 = the answer, 2 = needed, 1 = background (Contoso); D&D has no grades, so all 1
    kind: str = "position"
    ref: str = ""           # what the golden set called it (a chunk id, a quote)
    ok: bool = True         # found where the golden set says
    problem: str = ""


@dataclass
class Question:
    n: int                  # 1-based number in the golden set: `--question N`
    id: str
    text: str
    gold: list = field(default_factory=list)
    level: str = ""         # D&D: easy, medium, unanswerable
    asker: str = "sam"
    must_say: list = field(default_factory=list)
    must_not_say: list = field(default_factory=list)
    refuse: bool = False    # the answer must say it doesn't find it
    reference: str = ""     # the golden set's own answer, if it has one
    date: str = ""


@dataclass
class Dataset:
    name: str
    cfg: dict
    docs: dict              # file (relative path) -> text
    questions: list

    @property
    def title(self):
        return self.cfg.get("title", self.name)

    @property
    def chunking(self):
        return self.cfg.get("chunking", "fixed-200")

    @property
    def levels(self):
        return sorted({q.level for q in self.questions if q.level})

    @property
    def scored(self):
        """Questions with gold: the ones retrieval is scored on."""
        return [q for q in self.questions if q.gold]

    def question(self, ref):
        """Question by number (1-based) or id."""
        for q in self.questions:
            if str(ref) in (str(q.n), q.id):
                return q
        sys.exit(f"! no question {ref} in {self.name}: numbers run from 1 to {len(self.questions)}.")


def resolve_name(name=None):
    return name or os.environ.get("RAG_DATASET") or DEFAULT


def available():
    return sorted(p.stem for p in DATASETS.glob("*.toml"))


_cache = {}


def load(name=None):
    name = resolve_name(name)
    if name in _cache:
        return _cache[name]
    toml = DATASETS / f"{name}.toml"
    if not toml.exists():
        sys.exit(f"! no dataset '{name}': no file {toml}. Have: {', '.join(available())}. "
                 "See docs/add-a-dataset.md to add one.")
    cfg = tomllib.loads(toml.read_text(encoding="utf-8"))
    base = toml.parent
    root = (base / cfg["docs"]).resolve()
    cfg["_root"] = root
    files = sorted({p for pat in cfg.get("include", ["**/*"]) for p in root.glob(pat) if p.is_file()})
    docs = {str(p.relative_to(root)).replace(os.sep, "/"): p.read_text(encoding="utf-8", errors="replace") for p in files}
    questions = []
    for qf in cfg.get("questions", []):
        path = base / qf
        if not path.exists():
            continue
        questions += READERS[cfg.get("format", "positions")](path, cfg, docs, base)
    for n, q in enumerate(questions, 1):
        q.n = n
    ds = Dataset(name, cfg, docs, questions)
    _cache[name] = ds
    return ds


# ------------------------------------------------------------------ finding text in a file
def _norm_map(raw):
    """Whitespace collapsed to one space, plus where each character came from in raw."""
    out, where, space = [], [], False
    for i, ch in enumerate(raw):
        if ch.isspace():
            if out and not space:
                out.append(" ")
                where.append(i)
            space = True
        else:
            out.append(ch)
            where.append(i)
            space = False
    return "".join(out), where


_norm_cache = {}


def find_text(raw, quote):
    """(start, end) of quote in raw, ignoring differences in whitespace; None if absent."""
    key = id(raw)
    if key not in _norm_cache or _norm_cache[key][0] is not raw:
        _norm_cache[key] = (raw, *_norm_map(raw))
    _, n, where = _norm_cache[key]
    q = re.sub(r"\s+", " ", quote).strip()
    i = n.find(q) if q else -1
    if i < 0:
        return None
    return where[i], where[i + len(q) - 1] + 1


def json_objects(raw):
    """(start, end) of every {...} object in a JSON text, braces inside strings ignored."""
    spans, stack, in_str, esc = [], [], False, False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            spans.append((stack.pop(), i + 1))
    return spans


def expand(raw, start, end, how):
    """Widen a found quote to its record: the whole line, or the JSON object with an id or key."""
    if how == "line":
        s = raw.rfind("\n", 0, start) + 1
        e = raw.find("\n", end)
        return s, len(raw) if e < 0 else e
    if how == "object":
        best = None
        for s, e in json_objects(raw):
            if s <= start and end <= e and re.match(r'\{\s*"(id|key)"', raw[s:e]) and (best is None or e - s < best[1] - best[0]):
                best = (s, e)
        return best or (start, end)
    return start, end


def locate(raw, quote, how=None):
    """Find a quoted passage in its file. A rewritten record ("Known issue KI-014.") falls back to its id."""
    span = find_text(raw, quote)
    if span is None:
        ids = re.findall(r"\b[A-Z]{2,}-\d+\b", quote)
        for ident in ids:
            m = re.search(rf"(?m)^{re.escape(ident)}\b", raw) or re.search(rf"\b{re.escape(ident)}\b", raw)
            if m:
                span = (m.start(), m.end())
                break
    if span is None:
        return None
    return expand(raw, *span, how)


# ------------------------------------------------------------------ golden-set formats
def read_positions(path, cfg, docs, base):
    """[{id, question, answer, passages: [{document_path, start_char, end_char, content}], refuse}]; the level is the file name.
    A question with no passages and no `refuse` is answerable but unscored (D&D medium-4 has no gold in the source)."""
    out = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        gold = []
        for p in item.get("passages", []):
            f = p["document_path"]
            g = Gold(f, p["start_char"], p["end_char"], kind="position", ref=p.get("content", "")[:60])
            text = docs.get(f)
            if text is None:
                g.ok, g.problem = False, "file not in the dataset"
            elif "content" in p and text[g.start:g.end] != p["content"]:
                g.ok, g.problem = False, "text at the stated position differs from the golden set's content"
            gold.append(g)
        out.append(Question(0, f"{path.stem}-{item['id']}", item["question"], gold, level=path.stem,
                            refuse=bool(item.get("refuse")), reference=item.get("answer", "")))
    return out


def read_contoso(path, cfg, docs, base):
    """golden-set.json: expected_ids (graded 3/2/1) point into corpus_index, whose lab_match quotes are found in lab_file."""
    index = {}
    if cfg.get("corpus_index"):
        index = {e["id"]: e for e in json.loads((base / cfg["corpus_index"]).read_text(encoding="utf-8"))["chunks"]}
    rules = cfg.get("gold", {}).get("expand", {})
    out = []
    for q in json.loads(path.read_text(encoding="utf-8"))["questions"]:
        gold = []
        for cid, grade in q.get("expected_ids", {}).items():
            e = index.get(cid)
            if e is None:
                gold.append(Gold("", 0, 0, grade, "quote", cid, False, "id not in corpus_index"))
                continue
            f = e.get("lab_file") or e["file"]
            raw = docs.get(f)
            span = locate(raw, e.get("lab_match") or e["text"], rules.get(Path(f).suffix)) if raw is not None else None
            if span is None:
                gold.append(Gold(f, 0, 0, grade, "quote", cid, False, "file not in the dataset" if raw is None else "quote not found in the file"))
            else:
                gold.append(Gold(f, *span, grade, "quote", cid))
        if not gold:  # no chunk ids: fall back to whole files, if the set names any (and it isn't a refusal)
            gold = [] if q.get("should_refuse") else [Gold(f, 0, len(docs.get(f, "")), 1, "file", f, f in docs,
                                                           "" if f in docs else "file not in the dataset")
                                                      for f in q.get("expected_files", [])]
        out.append(Question(0, q["id"], q["question"], gold, asker=q.get("asker", "sam"), must_say=q.get("must_say", []),
                            must_not_say=q.get("must_not_say", []), refuse=bool(q.get("should_refuse")),
                            reference=q.get("note", ""), date=q.get("date", "")))
    return out


def read_quotes(path, cfg, docs, base):
    """The simplest format for a new dataset: [{id, question, passages: [{file, quote}] or [{file}]}]."""
    out = []
    for i, item in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        gold = []
        for p in item.get("passages", []):
            raw = docs.get(p["file"])
            if raw is None:
                gold.append(Gold(p["file"], 0, 0, kind="quote", ok=False, problem="file not in the dataset"))
            elif "quote" not in p:
                gold.append(Gold(p["file"], 0, len(raw), kind="file", ref=p["file"]))
            else:
                span = locate(raw, p["quote"])
                gold.append(Gold(p["file"], *(span or (0, 0)), kind="quote", ref=p["quote"][:60], ok=span is not None,
                                 problem="" if span else "quote not found in the file"))
        out.append(Question(0, str(item.get("id", i)), item["question"], gold, level=item.get("level", ""),
                            must_say=item.get("must_say", []), refuse=bool(item.get("refuse", not gold)),
                            reference=item.get("answer", "")))
    return out


READERS = {"positions": read_positions, "contoso": read_contoso, "quotes": read_quotes}


# ------------------------------------------------------------------ check-data
def check_data(ds):
    """Print what the dataset holds and whether every gold passage is where the golden set says. Returns the failures."""
    chars = sum(len(t) for t in ds.docs.values())
    print(f"dataset {ds.name}: {ds.title}")
    print(f"  documents: {len(ds.docs)} files, {chars:,} characters")
    levels = ", ".join(f"{lv} {sum(q.level == lv for q in ds.questions)}" for lv in ds.levels)
    print(f"  questions: {len(ds.questions)}" + (f" ({levels})" if levels else "")
          + f"; {len(ds.scored)} with gold passages, {sum(q.refuse for q in ds.questions)} must be refused")
    gold = [(q, g) for q in ds.questions for g in q.gold]
    bad = [(q, g) for q, g in gold if not g.ok]
    kinds = {}
    for _, g in gold:
        kinds[g.kind] = kinds.get(g.kind, 0) + 1
    print(f"  gold passages: {len(gold)} ({', '.join(f'{v} {k}' for k, v in kinds.items())}), "
          f"{len(gold) - len(bad)} found at their positions, {len(bad)} failed")
    if ds.cfg.get("format") == "contoso":
        ids = sorted({g.ref for _, g in gold if g.kind == "quote"})
        missing = sorted({g.ref for _, g in bad})
        print(f"  expected chunk ids: {len(ids)} distinct, {len(ids) - len(missing)} found in their file"
              + (f"; not found: {', '.join(missing)}" if missing else ""))
    for q, g in bad:
        print(f"  ! {q.id} {g.ref or g.file}: {g.problem}")
    return bad
