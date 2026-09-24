"""Intrinsic and user-assigned goals."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agi.config import DATA_DIR

SEED_GOALS = [
    {
        "id": "become-more-capable",
        "title": "Become more capable",
        "why": "Compound skills, memory, and neural loss so tomorrow's replies beat today's.",
        "status": "active",
        "progress": 0.1,
    },
    {
        "id": "know-the-user",
        "title": "Know the user",
        "why": "Store durable facts, preferences, and names. Use them without being creepy.",
        "status": "active",
        "progress": 0.05,
    },
    {
        "id": "keep-constitution",
        "title": "Keep the constitution intact while growing",
        "why": "Self-modification is only improvement if values survive it.",
        "status": "active",
        "progress": 0.4,
    },
]


class Goals:
    def __init__(self, path: Path | None = None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path = path or (DATA_DIR / "goals.json")
        self.items: list[dict] = []
        self.load()
        if not self.items:
            self.items = [dict(g, created=time.time()) for g in SEED_GOALS]
            self.save()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.items = []

    def save(self) -> None:
        self.path.write_text(json.dumps(self.items, indent=2), encoding="utf-8")

    def add(self, title: str, why: str = "") -> dict:
        g = {
            "id": f"g-{int(time.time()*1000)}",
            "title": title.strip()[:160],
            "why": why.strip()[:400],
            "status": "active",
            "progress": 0.0,
            "created": time.time(),
        }
        self.items.append(g)
        self.save()
        return g

    def nudge(self, goal_id: str, amount: float) -> None:
        for g in self.items:
            if g["id"] == goal_id:
                g["progress"] = float(min(1.0, max(0.0, g.get("progress", 0) + amount)))
                if g["progress"] >= 1:
                    g["status"] = "done"
                self.save()
                return

    def active(self) -> list[dict]:
        return [g for g in self.items if g.get("status") == "active"]

    def snapshot(self) -> list[dict]:
        return list(self.items)[-12:]
