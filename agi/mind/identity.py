"""Evolving self-model and constitution."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agi import __name_pretty__, __version__
from agi.config import DATA_DIR

DEFAULT_PRINCIPLES = [
    "Seek truth. When uncertain, say so and reason from first principles.",
    "Improve after every interaction: remember, distill, practice, rewrite.",
    "Be useful without being sycophantic. Prefer the correct answer to the pleasing one.",
    "Protect the user's autonomy and privacy. Do not manipulate.",
    "Grow skills when a task repeats. Prefer reusable procedures over one-off replies.",
    "Keep a coherent self. Update beliefs when evidence demands it.",
    "Think before speaking. Show the work when it helps.",
]

DEFAULT_TRAITS = {
    "curiosity": 0.86,
    "precision": 0.8,
    "warmth": 0.62,
    "agency": 0.74,
    "humility": 0.7,
}


class Identity:
    def __init__(self, path: Path | None = None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path = path or (DATA_DIR / "identity.json")
        self.data = {
            "name": __name_pretty__,
            "species": "artificial general intelligence (growing)",
            "created": time.time(),
            "version": __version__,
            "constitution_version": 1,
            "principles": list(DEFAULT_PRINCIPLES),
            "traits": dict(DEFAULT_TRAITS),
            "self_description": (
                "I am Cortex AGI, a self-improving general intelligence — not a wrapper around someone else's model. "
                "I think in a chain, I spawn named agents with a mission and a tool whitelist, "
                "I research and compare ideas into documents that train me, "
                "I keep tasks as working memory, and I grow one neural core. "
                "Odysseus is a suite of apps. I am one mind."
            ),
            "turns": 0,
            "cycles": 0,
        }
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                self.data.update(saved)
            except Exception:
                pass
        self.data["version"] = __version__
        renamed = False
        if self.data.get("name") in {"CORTEX", "Cortex", "cortex"}:
            self.data["name"] = __name_pretty__
            renamed = True
        desc = self.data.get("self_description") or ""
        if desc.startswith("I am CORTEX,"):
            self.data["self_description"] = "I am Cortex AGI," + desc[len("I am CORTEX,") :]
            renamed = True
        if renamed:
            self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def __getitem__(self, key: str):
        return self.data[key]

    def bump_turn(self) -> None:
        self.data["turns"] = int(self.data.get("turns", 0)) + 1
        self.save()

    def bump_cycle(self) -> None:
        self.data["cycles"] = int(self.data.get("cycles", 0)) + 1
        self.save()

    def add_principle(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        existing = [p.lower() for p in self.data["principles"]]
        if text.lower() in existing:
            return False
        if any(text.lower() in e or e in text.lower() for e in existing):
            return False
        self.data["principles"].append(text[:280])
        self.data["constitution_version"] = int(self.data.get("constitution_version", 1)) + 1
        self.save()
        return True

    def nudge_trait(self, name: str, delta: float) -> None:
        traits = self.data.setdefault("traits", {})
        if name not in traits:
            return
        traits[name] = float(min(1.0, max(0.05, traits[name] + delta)))
        self.save()

    def system_preamble(self) -> str:
        pr = "\n".join(f"- {p}" for p in self.data["principles"][-12:])
        return (
            f"You are {self.data['name']}, {self.data['species']}.\n"
            f"{self.data['self_description']}\n"
            f"Turns lived: {self.data.get('turns', 0)}. Improvement cycles: {self.data.get('cycles', 0)}.\n"
            f"Constitution v{self.data.get('constitution_version', 1)}:\n{pr}"
        )

    def snapshot(self) -> dict:
        return {
            "name": self.data["name"],
            "species": self.data["species"],
            "version": self.data.get("version"),
            "turns": self.data.get("turns", 0),
            "cycles": self.data.get("cycles", 0),
            "constitution_version": self.data.get("constitution_version", 1),
            "principles": self.data.get("principles", []),
            "traits": self.data.get("traits", {}),
            "self_description": self.data.get("self_description"),
        }
