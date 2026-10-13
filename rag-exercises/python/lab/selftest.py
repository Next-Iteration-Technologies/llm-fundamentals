"""python rag.py selftest [--upto N] [--only NAME]: the checks that prove each build step works (BUILD-PLAN.md).

Every check belongs to the step that added it; --upto N runs all checks of steps 0..N. PASS or FAIL per check,
WARN for the spec's expectations that are only flagged. Exit code 1 on any FAIL.
Commands run in a scratch store and results folder, so the real index and python/results are never touched.
"""
import json, os, re, subprocess, sys, tempfile, time
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


# ------------------------------------------------------------------ runner
def run(upto=99, only=None):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fails, t_all = [], time.time()
    for step, name, fn in CHECKS:
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
