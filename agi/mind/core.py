"""The AGI object: wires inference, memory, identity, goals, tools, improvement."""

from __future__ import annotations

import threading
from typing import Iterator

from agi.config import GGUF_PATH, MODEL_DIR, SAFETENSORS_PATH
from agi.improve.loop import SelfImprovement
from agi.inference.loader import ModelEngine, load_engine
from agi.mind.cognition import Cognition
from agi.mind.goals import Goals
from agi.mind.identity import Identity
from agi.mind.agent import get_roster
from agi.mind.desk import get_desk
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
        self.last_latency_ms = 0.0
        self.last_chain: dict | None = None
        self.engine.trainer.dream_source = self._dream_text

    def _dream_text(self) -> str:
        dlg = self.memory.recent_dialogue(k=12)
        facts = self.memory.all_facts(k=12)
        arts = self.knowledge.taught[-4:] or self.knowledge.search("agi", k=2)
        parts = [self.identity.system_preamble()]
        if dlg:
            parts.append("\n".join(f"{d['role']}: {d['content'][:240]}" for d in dlg))
        if facts:
            parts.append("; ".join(f"{f.subject} {f.predicate} {f.obj}" for f in facts))
        for a in arts:
            parts.append(f"{a.title}. {a.body}")
        return "\n".join(parts)[:4000]

    def think(self, user_text: str) -> Iterator[dict]:
        user_text = (user_text or "").strip()
        if not user_text:
            yield {"type": "done", "message": "Say something and I will think.", "intent": "empty", "thoughts": []}
            return
        self.memory.remember_episode("user", user_text)
        reply = ""
        intent = "chat"
        latency = 0.0
        for event in self.cognition.think(user_text):
            if event.get("type") == "done":
                reply = event.get("message") or ""
                intent = event.get("intent") or "chat"
                latency = float(event.get("latency_ms") or 0)
                self.last_latency_ms = latency
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
        latency = 0.0
        for event in self.think(user_text):
            if event.get("type") == "thought":
                thoughts.append(event)
            elif event.get("type") == "done":
                message = event.get("message") or ""
                intent = event.get("intent") or intent
                latency = float(event.get("latency_ms") or 0)
            elif event.get("type") == "improve":
                improve = event
        return {
            "message": message,
            "intent": intent,
            "thoughts": thoughts,
            "improve": improve,
            "latency_ms": latency,
            "chain": self.last_chain,
        }

    def quick_answer(self, user_text: str) -> str:
        return self.cognition.quick_answer(user_text)

    def stop(self) -> None:
        self.engine.request_stop()

    def teach(self, title: str, body: str) -> dict:
        art = self.knowledge.teach(title, body)
        self.memory.add_fact("knowledge", "taught", art.title, 0.95)
        self.engine.train_on(f"{art.title}. {art.body}", steps=6, blocking=False)
        return {"title": art.title, "body": art.body}

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
            "taught": [{"title": a.title, "tags": list(a.tags)} for a in self.knowledge.taught[-12:]],
            "latency_ms": self.last_latency_ms,
            "chain": self.last_chain,
            "desk": get_desk().counts(),
            "agents": get_roster().list(),
            "core": {
                "gguf": GGUF_PATH.exists(),
                "safetensors": SAFETENSORS_PATH.exists(),
                "model_dir": str(MODEL_DIR),
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
