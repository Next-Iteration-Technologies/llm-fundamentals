"""Everything slow is made once and kept on disk: embeddings, rerank scores, model replies.

One SQLite file (RAG_CACHE, default python/.store/lab-cache.sqlite). The key is a hash of everything that decides the
value (model, kind, text), so a changed text or model simply misses. Same input, same output: comparisons are
repeatable, and rerunning a comparison or an answer run costs nothing.
"""
import hashlib, json, os, sqlite3, time
from pathlib import Path

import numpy as np

from . import PY

_conn = None


def path():
    return Path(os.environ.get("RAG_CACHE") or PY / ".store" / "lab-cache.sqlite")


def db():
    global _conn
    if _conn is None:
        p = path()
        p.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(p), timeout=60)
        _conn.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, v BLOB)")
        _conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, v TEXT, saved TEXT)")
    return _conn


def key(*parts):
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def get_vecs(keys):
    out, c = {}, db()
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        for k, v in c.execute(f"SELECT key, v FROM vec WHERE key IN ({','.join('?' * len(part))})", part):
            out[k] = np.frombuffer(v, dtype=np.float32)
    return out


def put_vecs(pairs):
    c = db()
    c.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)", [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in pairs])
    c.commit()


def get(k):
    row = db().execute("SELECT v, saved FROM kv WHERE key = ?", (k,)).fetchone()
    return (json.loads(row[0]), row[1]) if row else (None, None)


def get_many(keys):
    out, c = {}, db()
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        for k, v in c.execute(f"SELECT key, v FROM kv WHERE key IN ({','.join('?' * len(part))})", part):
            out[k] = json.loads(v)
    return out


def put(k, value):
    c = db()
    c.execute("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", (k, json.dumps(value), time.strftime("%Y-%m-%d %H:%M")))
    c.commit()


def put_many(pairs):
    c, now = db(), time.strftime("%Y-%m-%d %H:%M")
    c.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?, ?)", [(k, json.dumps(v), now) for k, v in pairs])
    c.commit()
