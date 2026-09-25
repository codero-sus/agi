"""Episodic, semantic, and working memory with a numpy vector index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agi.config import DATA_DIR, MEMORY_INDEX_CAP

DIM = 96


def _embed(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    toks = [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]
    if not toks:
        toks = ["_empty"]
    for i, tok in enumerate(toks):
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        a = int.from_bytes(h[:4], "little")
        b = int.from_bytes(h[4:], "little")
        vec[a % DIM] += 1.0
        vec[b % DIM] -= 0.4
        vec[(a + i) % DIM] += 0.25
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-9 else vec


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


@dataclass
class Episode:
    id: int
    ts: float
    role: str
    content: str
    score: float = 0.0


@dataclass
class Fact:
    id: int
    ts: float
    subject: str
    predicate: str
    obj: str
    confidence: float


class Memory:
    def __init__(self, path: Path | None = None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path = path or (DATA_DIR / "memory.sqlite")
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-8000")
        self._lock = threading.Lock()
        self._init()
        self.working: list[dict] = []
        self._ids: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._mat: np.ndarray = np.zeros((0, DIM), dtype=np.float32)
        self._meta: list[tuple[float, str, str]] = []  # ts, role, content
        self._rebuild_index()

    def _init(self) -> None:
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                obj TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                text TEXT NOT NULL,
                source TEXT
            );
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ep_ts ON episodes(ts);
            CREATE INDEX IF NOT EXISTS idx_facts_s ON facts(subject);
            """
        )
        c.commit()

    def _rebuild_index(self) -> None:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, ts, role, content, embedding FROM episodes ORDER BY id DESC LIMIT ?",
                (MEMORY_INDEX_CAP,),
            ).fetchall()
        ids, mats, meta = [], [], []
        for rid, ts, role, content, emb in reversed(rows):
            if not emb:
                continue
            v = _from_blob(emb)
            if v.shape[0] != DIM:
                continue
            ids.append(rid)
            mats.append(v)
            meta.append((float(ts), role, content))
        if mats:
            self._ids = np.asarray(ids, dtype=np.int64)
            self._mat = np.stack(mats).astype(np.float32)
            self._meta = meta
        else:
            self._ids = np.zeros((0,), dtype=np.int64)
            self._mat = np.zeros((0, DIM), dtype=np.float32)
            self._meta = []

    def _index_append(self, rid: int, ts: float, role: str, content: str, vec: np.ndarray) -> None:
        self._ids = np.append(self._ids, np.int64(rid))
        self._mat = np.vstack([self._mat, vec.reshape(1, -1)]) if self._mat.size else vec.reshape(1, -1)
        self._meta.append((ts, role, content))
        if self._ids.shape[0] > MEMORY_INDEX_CAP:
            cut = self._ids.shape[0] - MEMORY_INDEX_CAP
            self._ids = self._ids[cut:]
            self._mat = self._mat[cut:]
            self._meta = self._meta[cut:]

    def remember_episode(self, role: str, content: str) -> int:
        vec = _embed(content)
        ts = time.time()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO episodes (ts, role, content, embedding) VALUES (?, ?, ?, ?)",
                (ts, role, content, _blob(vec)),
            )
            self.conn.commit()
            rid = int(cur.lastrowid)
        self._index_append(rid, ts, role, content, vec)
        self.working.append({"role": role, "content": content, "ts": ts})
        self.working = self.working[-24:]
        return rid

    def search(self, query: str, k: int = 5) -> list[Episode]:
        if self._mat.shape[0] == 0:
            return []
        q = _embed(query)
        scores = self._mat @ q
        k = min(k, scores.shape[0])
        # recent-item bonus so yesterday doesn't drown today
        recency = np.linspace(0.0, 0.08, scores.shape[0], dtype=np.float32)
        scores = scores + recency
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out: list[Episode] = []
        for i in idx:
            ts, role, content = self._meta[int(i)]
            out.append(Episode(int(self._ids[int(i)]), ts, role, content, float(scores[int(i)])))
        return out

    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 0.8) -> None:
        subject, predicate, obj = subject.strip()[:120], predicate.strip()[:80], obj.strip()[:400]
        if not subject or not obj:
            return
        with self._lock:
            existing = self.conn.execute(
                "SELECT id FROM facts WHERE subject=? AND predicate=? AND obj=?",
                (subject, predicate, obj),
            ).fetchone()
            if existing:
                self.conn.execute(
                    "UPDATE facts SET confidence=MIN(1.0, confidence+0.05), ts=? WHERE id=?",
                    (time.time(), existing[0]),
                )
            else:
                self.conn.execute(
                    "INSERT INTO facts (ts, subject, predicate, obj, confidence) VALUES (?, ?, ?, ?, ?)",
                    (time.time(), subject, predicate, obj, confidence),
                )
            self.conn.commit()

    def facts_about(self, query: str, k: int = 8) -> list[Fact]:
        q = query.lower()
        toks = [tok for tok in q.split() if len(tok) > 2]
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, ts, subject, predicate, obj, confidence FROM facts ORDER BY ts DESC LIMIT 200"
            ).fetchall()
        hits = []
        for r in rows:
            blob = f"{r[2]} {r[3]} {r[4]}".lower()
            if (toks and any(tok in blob for tok in toks)) or q in blob:
                hits.append(Fact(*r))
        return hits[:k]

    def all_facts(self, k: int = 40) -> list[Fact]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, ts, subject, predicate, obj, confidence FROM facts ORDER BY ts DESC LIMIT ?",
                (k,),
            ).fetchall()
        return [Fact(*r) for r in rows]

    def forget_facts(self, query: str) -> int:
        q = query.lower().strip()
        if not q:
            return 0
        with self._lock:
            rows = self.conn.execute("SELECT id, subject, predicate, obj FROM facts").fetchall()
            ids = [r[0] for r in rows if q in f"{r[1]} {r[2]} {r[3]}".lower()]
            for i in ids:
                self.conn.execute("DELETE FROM facts WHERE id=?", (i,))
            self.conn.commit()
        return len(ids)

    def add_lesson(self, text: str, source: str = "critic") -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            dup = self.conn.execute("SELECT id FROM lessons WHERE text=?", (text,)).fetchone()
            if dup:
                return
            self.conn.execute(
                "INSERT INTO lessons (ts, text, source) VALUES (?, ?, ?)",
                (time.time(), text[:500], source),
            )
            self.conn.commit()

    def lessons(self, k: int = 20) -> list[str]:
        with self._lock:
            rows = self.conn.execute("SELECT text FROM lessons ORDER BY id DESC LIMIT ?", (k,)).fetchall()
        return [r[0] for r in rows]

    def log_metric(self, name: str, value: float) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO metrics (ts, name, value) VALUES (?, ?, ?)",
                (time.time(), name, float(value)),
            )
            self.conn.commit()

    def metric_series(self, name: str, k: int = 80) -> list[tuple[float, float]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT ts, value FROM metrics WHERE name=? ORDER BY id DESC LIMIT ?",
                (name, k),
            ).fetchall()
        rows = list(reversed(rows))
        return [(float(a), float(b)) for a, b in rows]

    def log_event(self, kind: str, payload: dict) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
                (time.time(), kind, json.dumps(payload, ensure_ascii=False)[:2000]),
            )
            self.conn.commit()

    def recent_events(self, k: int = 30) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT ts, kind, payload FROM events ORDER BY id DESC LIMIT ?", (k,)
            ).fetchall()
        out = []
        for ts, kind, payload in rows:
            try:
                body = json.loads(payload)
            except Exception:
                body = {"raw": payload}
            out.append({"ts": ts, "kind": kind, "payload": body})
        return out

    def counts(self) -> dict:
        with self._lock:
            def n(table: str) -> int:
                return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

            return {
                "episodes": n("episodes"),
                "facts": n("facts"),
                "lessons": n("lessons"),
                "events": n("events"),
            }

    def recent_dialogue(self, k: int = 8) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content FROM episodes ORDER BY id DESC LIMIT ?", (k,)
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def history(self, k: int = 40) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT ts, role, content FROM episodes ORDER BY id DESC LIMIT ?", (k,)
            ).fetchall()
        return [{"ts": ts, "role": r, "content": c} for ts, r, c in reversed(rows)]

    def extract_from_user(self, text: str) -> list[tuple[str, str, str]]:
        import re

        t = text.strip()
        found: list[tuple[str, str, str]] = []
        patterns = [
            (r"\bmy name is ([A-Za-z][\w\s\-]{1,40})", "user", "name"),
            (r"\bi am (?:called|named) ([A-Za-z][\w\s\-]{1,40})", "user", "name"),
            (r"\bcall me ([A-Za-z][\w\s\-]{1,40})", "user", "name"),
            (r"\bi live in ([A-Za-z][\w\s,\-]{1,60})", "user", "lives_in"),
            (r"\bi work (?:as|at) ([^\.\,\n]{2,60})", "user", "works"),
            (r"\bi like ([^\.\,\n]{2,60})", "user", "likes"),
            (r"\bi love ([^\.\,\n]{2,60})", "user", "likes"),
            (r"\b(?:please )?remember that (.+)", "user", "note"),
            (r"\bdon't forget (?:that )?(.+)", "user", "note"),
            (r"\bmy favorite (\w+) is ([^\.\,\n]{1,60})", "user", "favorite"),
        ]
        for pat, subj, pred in patterns:
            m = re.search(pat, t, re.I)
            if m:
                if pred == "favorite" and len(m.groups()) >= 2:
                    found.append((subj, f"favorite_{m.group(1).lower()}", m.group(2).strip()))
                else:
                    found.append((subj, pred, m.group(1).strip().rstrip(".")))
        return found
