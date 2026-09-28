"""Documents and tasks the mind owns — not a separate app."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from agi.config import DATA_DIR, DOCS_DIR, ensure_dirs

_INDEX = "index.json"


class Desk:
    def __init__(self) -> None:
        ensure_dirs()
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self.docs_index = DOCS_DIR / _INDEX
        self.tasks_path = DATA_DIR / "tasks.json"
        self._lock = threading.Lock()
        self._docs: list[dict] = []
        self._tasks: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.docs_index.exists():
            try:
                self._docs = json.loads(self.docs_index.read_text(encoding="utf-8"))
            except Exception:
                self._docs = []
        if self.tasks_path.exists():
            try:
                self._tasks = json.loads(self.tasks_path.read_text(encoding="utf-8"))
            except Exception:
                self._tasks = []

    def _save_docs(self) -> None:
        self.docs_index.write_text(json.dumps(self._docs, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_tasks(self) -> None:
        self.tasks_path.write_text(json.dumps(self._tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_docs(self) -> list[dict]:
        with self._lock:
            return list(self._docs)

    def get_doc(self, did: str) -> dict | None:
        with self._lock:
            meta = next((d for d in self._docs if d["id"] == did), None)
            if not meta:
                return None
            path = DOCS_DIR / f"{did}.md"
            body = path.read_text(encoding="utf-8") if path.exists() else ""
            return {**meta, "body": body}

    def write_doc(self, title: str, body: str, kind: str = "note") -> dict:
        title = (title or "untitled").strip()[:120] or "untitled"
        body = (body or "").strip()[:80_000]
        kind = (kind or "note")[:24]
        did = uuid.uuid4().hex[:12]
        now = time.time()
        meta = {
            "id": did,
            "title": title,
            "kind": kind,
            "chars": len(body),
            "created": now,
            "updated": now,
        }
        with self._lock:
            (DOCS_DIR / f"{did}.md").write_text(body, encoding="utf-8")
            self._docs.insert(0, meta)
            self._docs = self._docs[:80]
            self._save_docs()
        return {**meta, "body": body}

    def save_doc(self, did: str, title: str | None = None, body: str | None = None) -> dict | None:
        with self._lock:
            meta = next((d for d in self._docs if d["id"] == did), None)
            if not meta:
                return None
            path = DOCS_DIR / f"{did}.md"
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if body is not None:
                current = body.strip()[:80_000]
                path.write_text(current, encoding="utf-8")
            if title:
                meta["title"] = title.strip()[:120]
            meta["chars"] = len(current)
            meta["updated"] = time.time()
            self._save_docs()
            return {**meta, "body": current}

    def delete_doc(self, did: str) -> bool:
        with self._lock:
            before = len(self._docs)
            self._docs = [d for d in self._docs if d["id"] != did]
            path = DOCS_DIR / f"{did}.md"
            if path.exists():
                path.unlink()
            if len(self._docs) != before:
                self._save_docs()
                return True
        return False

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return list(self._tasks)

    def add_task(self, title: str, source: str = "") -> dict:
        title = re.sub(r"\s+", " ", (title or "").strip())[:160]
        if not title:
            title = "untitled task"
        t = {
            "id": uuid.uuid4().hex[:10],
            "title": title,
            "done": False,
            "source": (source or "")[:80],
            "created": time.time(),
        }
        with self._lock:
            for existing in self._tasks:
                if not existing.get("done") and existing.get("title", "").lower() == title.lower():
                    return existing
            self._tasks.insert(0, t)
            self._tasks = self._tasks[:80]
            self._save_tasks()
        return t

    def patch_task(self, tid: str, done: bool | None = None, title: str | None = None) -> dict | None:
        with self._lock:
            for t in self._tasks:
                if t["id"] != tid:
                    continue
                if done is not None:
                    t["done"] = bool(done)
                if title:
                    t["title"] = title.strip()[:160]
                self._save_tasks()
                return t
        return None

    def delete_task(self, tid: str) -> bool:
        with self._lock:
            before = len(self._tasks)
            self._tasks = [t for t in self._tasks if t["id"] != tid]
            if len(self._tasks) != before:
                self._save_tasks()
                return True
        return False

    def counts(self) -> dict:
        with self._lock:
            open_n = sum(1 for t in self._tasks if not t.get("done"))
            return {"docs": len(self._docs), "tasks": len(self._tasks), "tasks_open": open_n}


_DESK: Desk | None = None
_DESK_LOCK = threading.Lock()


def get_desk() -> Desk:
    global _DESK
    with _DESK_LOCK:
        if _DESK is None:
            _DESK = Desk()
        return _DESK
