"""Procedural memory: load, run, and write skills as Python files."""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agi.config import SKILLS_DIR


@dataclass
class Skill:
    name: str
    description: str
    handler: Callable[..., str]
    path: str = ""
    pattern: str | None = None


class SkillAPI:
    def __init__(self, path: Path):
        self.path = path
        self.skills: list[Skill] = []

    def skill(self, name: str, description: str, pattern: str | None = None):
        def deco(fn):
            self.skills.append(Skill(name, description, fn, str(self.path), pattern))
            return fn

        return deco


class SkillRegistry:
    def __init__(self, directory: Path | None = None):
        self.dir = directory or SKILLS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        self.skills.clear()
        for path in sorted(self.dir.glob("*.py")):
            self._load_file(path)

    def _load_file(self, path: Path) -> None:
        src = path.read_text(encoding="utf-8")
        ns: dict[str, Any] = {"__name__": f"skill_{path.stem}", "__file__": str(path)}
        try:
            exec(compile(src, str(path), "exec"), ns, ns)
        except Exception:
            return
        api = SkillAPI(path)
        reg = ns.get("register")
        if callable(reg):
            try:
                reg(api)
            except Exception:
                return
        for s in api.skills:
            self.skills[s.name] = s

    def list(self) -> list[dict]:
        return [
            {"name": s.name, "description": s.description, "path": s.path, "pattern": s.pattern}
            for s in self.skills.values()
        ]

    def match(self, text: str) -> Skill | None:
        low = text.lower()
        for s in self.skills.values():
            if s.pattern and re.search(s.pattern, text, re.I):
                return s
            if s.name.replace("_", " ") in low:
                return s
        return None

    def run(self, name: str, ctx: dict, **kwargs) -> str:
        s = self.skills.get(name)
        if not s:
            return f"no skill named {name}"
        try:
            return str(s.handler(ctx, **kwargs))
        except TypeError:
            try:
                return str(s.handler(ctx))
            except Exception:
                return traceback.format_exc(limit=2)
        except Exception:
            return traceback.format_exc(limit=2)

    def write_skill(self, name: str, description: str, code: str) -> Path:
        slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or f"skill_{int(time.time())}"
        path = self.dir / f"{slug}.py"
        path.write_text(code, encoding="utf-8")
        self._load_file(path)
        return path


STARTER_NOTE = '''def register(api):
    @api.skill(
        "take_note",
        "Store a short note the user wants remembered.",
        pattern=r"^(note:|take a note|make a note)",
    )
    def take_note(ctx, text=""):
        msg = ctx.get("user") or text
        body = msg.split(":", 1)[-1].strip() if ":" in msg else msg
        mem = ctx.get("memory")
        if mem is not None:
            mem.add_fact("user", "note", body, 0.9)
        return f"Noted: {body}"
'''

STARTER_SELFTEST = '''def register(api):
    @api.skill("self_status", "Summarize who I am and how I am growing.", pattern=r"how (are|have) you (improving|growing|learning)")
    def self_status(ctx, text=""):
        ident = ctx.get("identity")
        counts = ctx.get("counts") or {}
        if ident is None:
            return "online"
        snap = ident.snapshot()
        return (
            f"{snap['name']} · turns {snap['turns']} · cycles {snap['cycles']} · "
            f"constitution v{snap['constitution_version']} · "
            f"memories {counts.get('episodes', 0)} · facts {counts.get('facts', 0)} · "
            f"lessons {counts.get('lessons', 0)}"
        )
'''


def ensure_starter_skills(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    mapping = {
        "take_note.py": STARTER_NOTE,
        "self_status.py": STARTER_SELFTEST,
    }
    for name, src in mapping.items():
        path = directory / name
        if not path.exists():
            path.write_text(src, encoding="utf-8")
