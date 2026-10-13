"""python rag.py selftest [--upto N] [--only NAME]: the checks that prove each build step works (BUILD-PLAN.md).

Every check belongs to the step that added it; --upto N runs all checks of steps 0..N. PASS or FAIL per check,
WARN for the spec's expectations that are only flagged. Exit code 1 on any FAIL.
Commands run in a scratch store and results folder, so the real index and python/results are never touched.
"""
import json, math, os, re, subprocess, sys, tempfile, time
from pathlib import Path

from . import PY, ROOT

BASELINE = ROOT / ".plan" / "baseline.json"
SCRATCH = Path(os.environ.get("SELFTEST_DIR") or Path(tempfile.gettempdir()) / "rag-selftest")
CHECKS = []          # (step, name, fn); fn returns None (pass) or raises Fail; may call warn()
WARNINGS = []


class Fail(Exception):
    pass


def check(step, name):
    def add(fn):
        CHECKS.append((step, name, fn))
        return fn
    return add


def warn(msg):
    WARNINGS.append(msg)
    print(f"    WARN {msg}")


def expect(cond, msg):
    if not cond:
        raise Fail(msg)


def env(**extra):
    """The environment for every command: no paid API, a scratch store and results folder, hash embedding unless asked."""
    (SCRATCH / "store").mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, LAB_NO_API="1", RAG_STORE=str(SCRATCH / "store"), RAG_RESULTS=str(SCRATCH / "results"),
             LLM_EMBED="hash", PYTHONIOENCODING="utf-8")
    e.update({k: str(v) for k, v in extra.items()})
    return e


def rag(*args, ok=True, timeout=3600, **extra):
    """Run `python rag.py ARGS`; returns stdout+stderr. ok=True: a non-zero exit is a failure."""
    t0 = time.time()
    p = subprocess.run([sys.executable, str(PY / "rag.py"), *args], cwd=PY, env=env(**extra), capture_output=True,
                       text=True, timeout=timeout)
    out = p.stdout + p.stderr
    if ok and p.returncode != 0:
        raise Fail(f"`rag.py {' '.join(args)}` exited {p.returncode}:\n" + "\n".join(out.strip().splitlines()[-12:]))
    if time.time() - t0 > 30:
        print(f"    ({time.time() - t0:.0f} s: rag.py {' '.join(args)})")
    return out if ok else (p.returncode, out)


# ------------------------------------------------------------------ check 1: the frozen starting point
def legacy_numbers():
    """Today's Contoso numbers on the hashed embedding: chunk counts, and eval --dry --no-rerank per question."""
    counts = {}
    for flag in ("", "--naive", "--children"):
        out = rag("ingest", "--dry", "--embed", "hash", *([flag] if flag else []))
        counts[flag or "structure"] = int(re.search(r"ingested (\d+) chunks", out).group(1))
    rag("ingest", "--dry", "--embed", "hash")  # eval reads the structure-aware collection
    out = rag("eval", "--dry", "--no-rerank", "--embed", "hash", RAG_DATASET="")
    per_q = {}
    for line in out.splitlines():
        m = re.match(r"id=(\S+)", line)
        if m:
            per_q[m.group(1)] = {k: float(v) for k, v in re.findall(r"(R@5|MRR|nDCG@5)=([\d.]+)", line)}
    means = {k: float(v) for k, v in re.findall(r"mean (R@5|MRR|nDCG@5): ([\d.]+)", out)}
    return {"chunks": counts, "means": means, "per_question": per_q}


def save_baseline():
    BASELINE.parent.mkdir(exist_ok=True)
    nums = legacy_numbers()
    BASELINE.write_text(json.dumps({"saved": time.strftime("%Y-%m-%d %H:%M"), "embed": "hash", **nums}, indent=1) + "\n")
    print(f"saved {BASELINE}: chunks {nums['chunks']}, means {nums['means']}")


@check(0, "frozen starting point: Contoso chunk counts and eval --dry --no-rerank unchanged")
def _frozen():
    expect(BASELINE.exists(), f"{BASELINE} is missing: run `python rag.py selftest --save-baseline` on the starting commit")
    want = json.loads(BASELINE.read_text())
    got = legacy_numbers()
    for k in ("chunks", "means", "per_question"):
        expect(got[k] == want[k], f"{k} changed: was {want[k]}, now {got[k]}")


# ------------------------------------------------------------------ check 3: the data is right
@check(1, "data: all 264 D&D gold passages at their positions; every Contoso expected chunk id found in its file")
def _data():
    out = rag("check-data", "--dataset", "dnd-srd")
    expect("gold passages: 264 (264 position), 264 found at their positions, 0 failed" in out, out)
    expect("(easy 25, medium 31, unanswerable 4)" in out, "D&D should have 56 golden questions plus 4 unanswerable:\n" + out)
    out = rag("check-data", RAG_DATASET="")
    expect("dataset contoso" in out and "questions: 18;" in out, "the default dataset should be contoso, 18 questions:\n" + out)
    m = re.search(r"expected chunk ids: (\d+) distinct, (\d+) found", out)
    expect(m and m.group(1) == m.group(2) and "not found" not in out, out)
    out = rag("check-data", RAG_DATASET="dnd-srd")
    expect("dataset dnd-srd" in out, "RAG_DATASET=dnd-srd should pick the rulebook:\n" + out)


# ------------------------------------------------------------------ step 2: the walkthrough, model choice, no paid API
STEPS8 = ["1. LOAD", "2. CHUNK", "3. EMBED", "4. SEARCH", "5. RERANK", "6. PROMPT", "7. ANSWER", "8. CHECK"]


@check(2, "walkthrough --dry: steps 1-5, then stops and says so (Contoso and D&D)")
def _walk_dry():
    for args in (["Why did we drop vendor A?"], ["Which race best resists a Red Dragon's breath?", "--dataset", "dnd-srd"]):
        out = rag("walkthrough", *args, "--dry", RAG_DATASET="")
        for s in STEPS8[:5]:
            expect(s in out, f"step '{s}' missing in walkthrough {args}:\n{out[-1500:]}")
        expect("--dry: stopping after step 5" in out and "6. PROMPT" not in out, "--dry should stop after rerank and say so")


@check(2, "walkthrough --question 1 (D&D, local model): all eight steps, prompt size, the model named, gold found/not found")
def _walk_full():
    out = rag("walkthrough", "--question", "1", RAG_DATASET="dnd-srd")
    for s in STEPS8:
        expect(s in out, f"step '{s}' missing:\n{out[-2000:]}")
    expect(re.search(r"6\. PROMPT: exactly what the model is sent: about [\d,]+ tokens", out), "the prompt step must show its size")
    expect(re.search(r"7\. ANSWER: \S+ \(.+\) answered", out), "the answer step must name the model")
    expect(re.search(r"found \(rank \d+\)|not found", out), "the check step must list found (rank N) / not found")
    out = rag("walkthrough", "Which race best resists a Red Dragon's breath?", "--dataset", "dnd-srd")
    expect("8. CHECK: skipped" in out, "a typed question must skip the check step")


@check(2, "LAB_NO_API: a paid block stops with the message and sends nothing; check reports it")
def _no_api():
    code, out = rag("walkthrough", "--question", "1", "--chat", "sonnet", ok=False, RAG_DATASET="dnd-srd")
    expect(code != 0 and "LAB_NO_API is set: this run may only use local models." in out, f"exit {code}:\n{out[-800:]}")
    for m in ("sonnet", "opus"):
        out = rag("check", "--chat", m)
        expect("LAB_NO_API is set" in out, f"check --chat {m} should report LAB_NO_API:\n{out}")


@check(2, "a prompt too big for Ollama's window stops with a clear message instead of being cut")
def _window():
    code, out = rag("walkthrough", "--question", "1", "--k", "60", "--chat", "ollama", ok=False, RAG_DATASET="dnd-srd")
    expect(code != 0 and "would cut it silently" in out and "window" in out, f"exit {code}:\n{out[-800:]}")


# ------------------------------------------------------------------ step 3: scores, perfect and random, saved runs
@check(3, "scores computed right: hand-made cases worked out on paper")
def _scores():
    from .metrics import score, matches
    from .datasets import Gold
    from .pipeline import rrf
    s = score([2], [1])                     # five results, the first right one at rank 2
    expect(abs(s["MRR"] - 0.5) < 1e-9, f"MRR with the first right chunk at rank 2 should be 0.5, got {s['MRR']}")
    s = score([1, 3, 7], [1, 1, 1])         # two of three gold passages in the top 3
    expect(abs(s["R@3"] - 2 / 3) < 1e-9, f"Recall@3 with 2 of 3 in the top 3 should be 0.67, got {s['R@3']}")
    fused = dict(rrf([["a", "b"], ["b", "a"]]))  # the deck: ranks 1 and 2 give 1/61 + 1/62
    expect(abs(fused["a"] - (1 / 61 + 1 / 62)) < 1e-12 and round(fused["a"], 4) == 0.0325, f"RRF: {fused}")
    g = Gold("f", 0, 100)
    for ov, want in ((49, False), (50, True), (51, True)):  # half of the smaller (the gold passage, 100)
        c = {"file": "f", "start": 100 - ov, "end": 300}
        expect(matches(c, g) == want, f"overlap {ov}% of the smaller should {'' if want else 'not '}match")
    expect(not matches({"file": "other", "start": 0, "end": 100}, g), "a chunk from another file never matches")
    s = score([1, None], [3, 1])
    expect(s["all@5"] == 0.0 and s["R@5"] == 0.5 and abs(s["nDCG@5"] - 3 / (3 + 1 / math.log2(3))) < 1e-9, f"{s}")


@check(3, "the test tells good from bad: perfect = 1.00 everywhere, random below 0.10 (both datasets)")
def _perfect_random():
    for ds in ("dnd-srd", "contoso"):
        out = rag("eval", "--dataset", ds, "--retriever", "perfect", "--dry")
        m = re.search(r"Recall@3 ([\d.]+)\s+Recall@5 ([\d.]+)\s+MRR ([\d.]+)\s+nDCG@5 ([\d.]+)\s+all found ([\d.]+)", out)
        expect(m and all(float(x) == 1.0 for x in m.groups()), f"perfect on {ds} is not 1.00 everywhere:\n{out[-900:]}")
        out = rag("eval", "--dataset", ds, "--retriever", "random", "--dry")
        m = re.search(r"Recall@3 ([\d.]+)\s+Recall@5 ([\d.]+)\s+MRR ([\d.]+)\s+nDCG@5 ([\d.]+)", out)
        expect(m and all(float(x) < 0.10 for x in m.groups()), f"random on {ds} is not below 0.10:\n{out[-900:]}")


@check(3, "eval --save baseline writes a run; report shows it with easy and medium apart")
def _baseline_report():
    out = rag("eval", "--dataset", "dnd-srd", "--dry", "--save", "baseline")
    expect("saved as eval/baseline" in out and "easy" in out and "medium" in out, out[-800:])
    r = json.loads((SCRATCH / "results" / "dnd-srd" / "eval" / "baseline.json").read_text())
    expect(r["summary"]["n"] == 55 and len(r["questions"]) == 55, "the D&D baseline should score 55 questions")
    out = rag("report")
    text = (SCRATCH / "results" / "RESULTS.md").read_text()
    expect("## dnd-srd" in text and "| baseline |" in text and "By level" in text and "easy" in text and "medium" in text,
           "RESULTS.md should show the baseline, easy and medium apart:\n" + text[:1500])


# ------------------------------------------------------------------ step 4: chunking
CHANGED_ON_PURPOSE = {"ki", "mail", "ticket", "jira"}   # rows with column names written in, mail without quotes, records


@check(4, "data: every chunk's position matches its file, for every chunking (text changed on purpose only where meant)")
def _positions():
    os.environ.setdefault("RAG_CACHE", str(PY / ".store" / "lab-cache.sqlite"))
    from . import datasets, chunkers
    for name in ("dnd-srd", "contoso"):
        ds = datasets.load(name)
        names = ["fixed-200", "naive-500", "recursive", "sentence", "semantic", "hierarchical"] + (["structure"] if name == "contoso" else [])
        for ch in names:
            for c in chunkers.chunk(ds, ch, "hash") + ([] if ch != "hierarchical" else chunkers.units(ds, ch, "hash")):
                raw = ds.docs[c["file"]]
                expect(0 <= c["start"] < c["end"] <= len(raw), f"{name} {ch}: bad span {c['file']} {c['start']}-{c['end']}")
                piece = raw[c["start"]:c["end"]]
                if c.get("type") in CHANGED_ON_PURPOSE:
                    ident = re.search(r"\b(KI-\d+|TCK-\d+|POC-\d+)\b", c["text"])
                    expect(c["type"] == "mail" or (ident and ident.group(1) in piece), f"{name} {ch}: {c['source']} not at its record")
                elif c.get("type"):  # structure-aware sections, config and log lines: the text is the section, normalized
                    flat = re.sub(r"\s+", " ", re.sub(r"<!--.*?-->", "", piece, flags=re.S))
                    expect(c["text"] in flat, f"{name} {ch}: {c['source']} text is not at its position")
                else:
                    expect(c["text"] == piece, f"{name} {ch}: chunk {c['id']} text differs from {c['file']} {c['start']}-{c['end']}")


@check(4, "chunks: count, sizes and gold passages cut in two, per chunking (D&D)")
def _chunks():
    out = rag("chunks", "--dataset", "dnd-srd")
    for ch in ("fixed-200", "recursive", "sentence", "semantic", "hierarchical"):
        expect(re.search(rf"^\s+{ch}\s+[\d,]+\s+\d+\s+\d+\s+\d+\s+\d+ of 264", out, re.M), f"no line for {ch}:\n{out}")


def compare_twice(*args):
    """Run a comparison twice into two folders: the saved runs must be identical apart from date and took_s."""
    outs = []
    for n in (1, 2):
        res = SCRATCH / f"twice{n}"
        rag("compare", *args, RAG_RESULTS=res)
        outs.append({p.relative_to(res).as_posix(): json.loads(p.read_text()) for p in sorted(res.rglob("*.json"))})
    expect(outs[0].keys() == outs[1].keys(), f"different runs saved: {outs[0].keys()} vs {outs[1].keys()}")
    for k in outs[0]:
        a, b = ({x: y for x, y in r.items() if x not in ("date", "took_s")} for r in (outs[0][k], outs[1][k]))
        expect(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), f"{k} differs between two identical runs")
    return outs[0]


@check(4, "same input, same output: compare --phase chunking twice gives identical runs (D&D and Contoso)")
def _twice_chunking():
    compare_twice("--phase", "chunking", "--dataset", "dnd-srd", "--dry")
    runs = compare_twice("--phase", "chunking", "--dataset", "contoso", "--dry")
    r5 = {k.split("/")[-1][:-5]: v["summary"]["R@5"] for k, v in runs.items()}
    fair = {k.split("/")[-1][:-5]: v["summary"]["R@2k"] for k, v in runs.items()}
    if not r5["naive-500"] < r5["structure"]:
        warn(f"Contoso (hash): naive-500 is not below structure-aware on Recall@5 ({r5['naive-500']:.2f} vs {r5['structure']:.2f}); "
             f"same-size view R@~2k: naive {fair['naive-500']:.2f}, structure {fair['structure']:.2f}")


@check(4, "show: the gold text and what was found, next to each other, overlap marked")
def _show():
    rag("compare", "--phase", "chunking", "--dataset", "dnd-srd", "--dry", "--rows", "fixed-200")
    out = rag("show", "3", "--run", "chunking/fixed-200", "--dataset", "dnd-srd")
    expect("GOLD 1 of" in out and ("FOUND rank" in out or "No chunk in the top 10" in out), out[-1500:])
    expect("»" in out or "No chunk in the top 10" in out, "the overlap should be marked:\n" + out[-1500:])


@check(4, "contextual chunking: the local model writes a context sentence for 10 Contoso chunks")
def _contextual():
    out = rag("chunks", "--dataset", "contoso", "--chunking", "contextual", "--limit", "10", "--show", "10", timeout=7200)
    n = out.count("    context: ")
    expect(n >= 5, f"expected context sentences for the first chunks, got {n}:\n{out[-2000:]}")


# ------------------------------------------------------------------ check 7: the student story works
CHAPTERS_BY_STEP = {2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6], 8: [7], 9: [8, 10], 10: [9]}
LOOK_FOR = {1: STEPS8, 2: ["Recall@3", "MRR", "PERFECT student", "RANDOM guesser", "questions with gold passages"],
            3: ["GOLD 1 of", "»", "cut in two", "COMPARE chunking", "won"]}


def chapter_commands(n):
    text = (ROOT / "LAB.md").read_text(encoding="utf-8")
    m = re.search(rf"^## Chapter {n}\..*?(?=^## |\Z)", text, re.S | re.M)
    expect(m, f"LAB.md has no Chapter {n}")
    cmds = []
    for block in re.findall(r"```\n(.*?)```", m.group(0), re.S):
        for line in block.splitlines():
            line = line.split("  #")[0].strip()
            if line.startswith("python rag.py"):
                import shlex
                cmds.append(shlex.split(line)[2:])
    return cmds


def student_story(step):
    def fn():
        for n in CHAPTERS_BY_STEP[step]:
            cmds = chapter_commands(n)
            expect(cmds, f"Chapter {n} has no commands")
            for ds in ("dnd-srd", "contoso"):
                outs = [rag(*c, RAG_DATASET=ds) for c in cmds]
                for term in LOOK_FOR.get(n, []):
                    expect(any(term in o for o in outs), f"Chapter {n} ({ds}): '{term}' is not in the output of its commands")
    return fn


# ------------------------------------------------------------------ runner
for _step, _chs in CHAPTERS_BY_STEP.items():
    check(_step, f"student story: LAB.md chapter {', '.join(map(str, _chs))} runs on dnd-srd and contoso")(student_story(_step))


def run(upto=99, only=None):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fails, t_all = [], time.time()
    for step, name, fn in sorted(CHECKS, key=lambda c: c[0]):
        if step > upto or (only and only not in name):
            continue
        t0 = time.time()
        try:
            fn()
            print(f"PASS [step {step}] {name}  ({time.time() - t0:.0f} s)")
        except Fail as e:
            fails.append(name)
            print(f"FAIL [step {step}] {name}\n    " + str(e).replace("\n", "\n    "))
        except Exception as e:
            fails.append(name)
            print(f"FAIL [step {step}] {name}\n    {type(e).__name__}: {e}")
        sys.stdout.flush()
    print(f"\n{'FAILED' if fails else 'ALL PASSED'}: {len(fails)} failed, {len(WARNINGS)} warnings, {time.time() - t_all:.0f} s")
    return 1 if fails else 0
