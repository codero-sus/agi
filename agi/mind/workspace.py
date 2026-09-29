"""Chats, a file vault, and prompt presets — the workspace Open WebUI charges for."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from agi.config import DATA_DIR, VAULT_DIR, ensure_dirs

TEXT_EXT = {".txt", ".md", ".py", ".json", ".csv", ".html", ".css", ".js", ".ts", ".toml", ".yml", ".yaml", ".rst", ".ini"}
MAX_FILE = 400_000

PROMPTS = [
    {"id": "research", "title": "Research", "body": "Research this: "},
    {"id": "do", "title": "Do", "body": "Do: "},
    {"id": "spawn", "title": "Spawn agent", "body": "create agent Name — mission: "},
    {"id": "runagent", "title": "Run agent", "body": "run Operator: "},
    {"id": "why", "title": "Explain why", "body": "Why does this work, in mechanism not slogans:\n"},
    {"id": "compare", "title": "Compare", "body": "Compare A and B on primitives, domain, and failure modes:\n"},
    {"id": "plan", "title": "Plan", "body": "Plan this as goal → gap → next action:\n"},
    {"id": "note", "title": "Note", "body": "Note: "},
    {"id": "todo", "title": "Todo", "body": "Todo: "},
    {"id": "review", "title": "Review code", "body": "Review this code. Bugs, complexity, tests:\n\n```\n\n```"},
    {"id": "teach", "title": "Teach CORTEX", "body": "Learn this: Title — "},
    {"id": "improve", "title": "Self-improve", "body": "Improve yourself"},
]


class Workspace:
    def __init__(self):
        ensure_dirs()
        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = DATA_DIR / "sessions.json"
        self._lock = threading.Lock()
        self.data = {"sessions": []}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                self.data.setdefault("sessions", [])
            except Exception:
                self.data = {"sessions": []}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list(self) -> list[dict]:
        with self._lock:
            out = []
            for s in self.data["sessions"]:
                msgs = s.get("messages") or []
                out.append(
                    {
                        "id": s["id"],
                        "title": s.get("title") or "untitled",
                        "updated": s.get("updated"),
                        "created": s.get("created"),
                        "pinned": bool(s.get("pinned")),
                        "count": len(msgs),
                        "preview": (msgs[-1]["content"][:80] if msgs else ""),
                    }
                )
            out.sort(key=lambda x: (not x["pinned"], -(x["updated"] or 0)))
            return out

    def create(self, title: str = "new thread") -> dict:
        s = {
            "id": uuid.uuid4().hex[:12],
            "title": (title or "new thread").strip()[:80],
            "created": time.time(),
            "updated": time.time(),
            "pinned": False,
            "messages": [],
        }
        with self._lock:
            self.data["sessions"].insert(0, s)
            self._save()
        return {"id": s["id"], "title": s["title"], "updated": s["updated"], "pinned": False, "count": 0, "preview": ""}

    def get(self, sid: str) -> dict | None:
        with self._lock:
            for s in self.data["sessions"]:
                if s["id"] == sid:
                    return s
        return None

    def append(self, sid: str, role: str, content: str, meta: dict | None = None) -> None:
        with self._lock:
            for s in self.data["sessions"]:
                if s["id"] != sid:
                    continue
                s.setdefault("messages", []).append(
                    {
                        "id": uuid.uuid4().hex[:10],
                        "role": role,
                        "content": content,
                        "ts": time.time(),
                        "meta": meta or {},
                    }
                )
                s["updated"] = time.time()
                if s.get("title") in ("new thread", "untitled") and role == "user":
                    s["title"] = re.sub(r"\s+", " ", content).strip()[:48] or s["title"]
                self._save()
                return

    def patch(self, sid: str, **fields) -> dict | None:
        with self._lock:
            for s in self.data["sessions"]:
                if s["id"] != sid:
                    continue
                if fields.get("title"):
                    s["title"] = str(fields["title"]).strip()[:80]
                if fields.get("pinned") is not None:
                    s["pinned"] = bool(fields["pinned"])
                s["updated"] = time.time()
                self._save()
                return s
        return None

    def delete(self, sid: str) -> bool:
        with self._lock:
            before = len(self.data["sessions"])
            self.data["sessions"] = [s for s in self.data["sessions"] if s["id"] != sid]
            if len(self.data["sessions"]) != before:
                self._save()
                return True
        return False

    def ingest(self, filename: str, data: bytes) -> dict:
        name = Path(filename).name
        ext = Path(name).suffix.lower()
        if ext not in TEXT_EXT:
            return {"ok": False, "error": f"unsupported type {ext or '(none)'}. text/code only."}
        if len(data) > MAX_FILE:
            return {"ok": False, "error": "file too large (400KB cap)."}
        text = data.decode("utf-8", errors="replace")
        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        dest = VAULT_DIR / f"{uuid.uuid4().hex[:8]}_{name}"
        dest.write_text(text, encoding="utf-8")
        excerpt = text[:6000]
        return {
            "ok": True,
            "name": name,
            "path": dest.name,
            "bytes": len(data),
            "chars": len(text),
            "excerpt": excerpt,
        }

    def vault(self) -> list[dict]:
        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        items = []
        for p in sorted(VAULT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file():
                items.append({"name": p.name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime})
        return items[:80]

    def read_vault(self, name: str) -> str | None:
        p = VAULT_DIR / Path(name).name
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")[:20000]
        return None

    def search_vault(self, query: str, k: int = 4) -> list[dict]:
        toks = [t for t in re.split(r"\W+", (query or "").lower()) if len(t) > 2]
        if not toks:
            return []
        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        scored: list[tuple[float, dict]] = []
        for p in VAULT_DIR.iterdir():
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")[:20000]
            except Exception:
                continue
            hay = (p.name + " " + text).lower()
            score = sum(1.0 for t in toks if t in hay)
            if p.stem.lower() in (query or "").lower():
                score += 3
            if score:
                idx = hay.find(toks[0])
                start = max(0, idx - 80)
                excerpt = text[start : start + 500]
                scored.append((score, {"name": p.name, "score": score, "excerpt": excerpt, "chars": len(text)}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]


_WS: Workspace | None = None
_WS_LOCK = threading.Lock()


def get_workspace() -> Workspace:
    global _WS
    with _WS_LOCK:
        if _WS is None:
            _WS = Workspace()
        return _WS
