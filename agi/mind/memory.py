"""Episodic, semantic, and working memory with hashed embeddings."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agi.config import DATA_DIR

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
        self._init()
        self.working: list[dict] = []

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

    def remember_episode(self, role: str, content: str) -> int:
        vec = _embed(content)
        cur = self.conn.execute(
            "INSERT INTO episodes (ts, role, content, embedding) VALUES (?, ?, ?, ?)",
            (time.time(), role, content, _blob(vec)),
        )
        self.conn.commit()
        self.working.append({"role": role, "content": content, "ts": time.time()})
        self.working = self.working[-24:]
        return int(cur.lastrowid)

    def search(self, query: str, k: int = 5) -> list[Episode]:
        q = _embed(query)
        rows = self.conn.execute("SELECT id, ts, role, content, embedding FROM episodes ORDER BY id DESC LIMIT 400").fetchall()
        scored: list[Episode] = []
        for rid, ts, role, content, emb in rows:
            if not emb:
                continue
            v = _from_blob(emb)
            if v.shape[0] != q.shape[0]:
                continue
            score = float(np.dot(q, v))
            scored.append(Episode(rid, ts, role, content, score))
        scored.sort(key=lambda e: e.score, reverse=True)
        return scored[:k]

    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 0.8) -> None:
        subject, predicate, obj = subject.strip()[:120], predicate.strip()[:80], obj.strip()[:400]
        if not subject or not obj:
            return
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
        rows = self.conn.execute(
            "SELECT id, ts, subject, predicate, obj, confidence FROM facts ORDER BY ts DESC LIMIT 200"
        ).fetchall()
        hits = []
        for r in rows:
            blob = f"{r[2]} {r[3]} {r[4]}".lower()
            if any(tok in blob for tok in q.split() if len(tok) > 2) or q in blob:
                hits.append(Fact(*r))
        return hits[:k]

    def all_facts(self, k: int = 40) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT id, ts, subject, predicate, obj, confidence FROM facts ORDER BY ts DESC LIMIT ?",
            (k,),
        ).fetchall()
        return [Fact(*r) for r in rows]

    def add_lesson(self, text: str, source: str = "critic") -> None:
        text = text.strip()
        if not text:
            return
        dup = self.conn.execute("SELECT id FROM lessons WHERE text=?", (text,)).fetchone()
        if dup:
            return
        self.conn.execute(
            "INSERT INTO lessons (ts, text, source) VALUES (?, ?, ?)",
            (time.time(), text[:500], source),
        )
        self.conn.commit()

    def lessons(self, k: int = 20) -> list[str]:
        rows = self.conn.execute("SELECT text FROM lessons ORDER BY id DESC LIMIT ?", (k,)).fetchall()
        return [r[0] for r in rows]

    def log_metric(self, name: str, value: float) -> None:
        self.conn.execute(
            "INSERT INTO metrics (ts, name, value) VALUES (?, ?, ?)",
            (time.time(), name, float(value)),
        )
        self.conn.commit()

    def metric_series(self, name: str, k: int = 80) -> list[tuple[float, float]]:
        rows = self.conn.execute(
            "SELECT ts, value FROM metrics WHERE name=? ORDER BY id ASC LIMIT ?",
            (name, k),
        ).fetchall()
        return [(float(a), float(b)) for a, b in rows]

    def log_event(self, kind: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, ensure_ascii=False)[:2000]),
        )
        self.conn.commit()

    def recent_events(self, k: int = 30) -> list[dict]:
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
        def n(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        return {
            "episodes": n("episodes"),
            "facts": n("facts"),
            "lessons": n("lessons"),
            "events": n("events"),
        }

    def recent_dialogue(self, k: int = 8) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM episodes ORDER BY id DESC LIMIT ?", (k,)
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def extract_from_user(self, text: str) -> list[tuple[str, str, str]]:
        """Pull obvious autobiographical facts from user speech."""
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
