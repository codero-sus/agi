"""The AGI object: wires inference, memory, identity, goals, tools, improvement."""

from __future__ import annotations

import threading
from typing import Iterator

from agi.improve.loop import SelfImprovement
from agi.inference.loader import ModelEngine, load_engine
from agi.mind.cognition import Cognition
from agi.mind.goals import Goals
from agi.mind.identity import Identity
from agi.mind.knowledge import Knowledge
from agi.mind.memory import Memory
from agi.tools.builtin import ToolRegistry


class AGI:
    def __init__(self, engine: ModelEngine | None = None):
        self.engine = engine or load_engine()
        self.identity = Identity()
        self.memory = Memory()
        self.knowledge = Knowledge()
        self.goals = Goals()
        self.tools = ToolRegistry()
        self.improver = SelfImprovement(self)
        self.cognition = Cognition(self)
        self.lock = threading.Lock()

    def think(self, user_text: str) -> Iterator[dict]:
        user_text = (user_text or "").strip()
        if not user_text:
            yield {"type": "done", "message": "Say something and I will think.", "intent": "empty", "thoughts": []}
            return
        self.memory.remember_episode("user", user_text)
        reply = ""
        intent = "chat"
        for event in self.cognition.think(user_text):
            if event.get("type") == "done":
                reply = event.get("message") or ""
                intent = event.get("intent") or "chat"
            yield event
        if reply:
            self.memory.remember_episode("agi", reply)
        improve = self.improver.after_turn(user_text, reply, intent)
        yield {"type": "improve", **improve}

    def chat(self, user_text: str) -> dict:
        message = ""
        thoughts = []
        intent = "chat"
        improve = {}
        for event in self.think(user_text):
            if event.get("type") == "thought":
                thoughts.append(event)
            elif event.get("type") == "done":
                message = event.get("message") or ""
                intent = event.get("intent") or intent
            elif event.get("type") == "improve":
                improve = event
        return {"message": message, "intent": intent, "thoughts": thoughts, "improve": improve}

    def quick_answer(self, user_text: str) -> str:
        return self.cognition.quick_answer(user_text)

    def state(self) -> dict:
        return {
            "identity": self.identity.snapshot(),
            "engine": self.engine.info.as_dict(),
            "neural": self.engine.neural_stats(),
            "memory": self.memory.counts(),
            "facts": [
                {
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.obj,
                    "confidence": f.confidence,
                }
                for f in self.memory.all_facts(k=24)
            ],
            "lessons": self.memory.lessons(k=16),
            "goals": self.goals.snapshot(),
            "skills": self.improver.skills.list(),
            "tools": self.tools.names(),
            "events": self.memory.recent_events(k=24),
            "metrics": {
                "loss": self.memory.metric_series("loss", 80),
                "self_eval": self.memory.metric_series("self_eval", 40),
            },
        }


_SINGLETON: AGI | None = None
_SINGLETON_LOCK = threading.Lock()


def get_agi() -> AGI:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = AGI()
        return _SINGLETON
