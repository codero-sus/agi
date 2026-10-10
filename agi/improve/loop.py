"""The self-improvement loop.

Light pass: after every turn — facts, critique, queue gradient steps (non-blocking).
Medium pass: every N turns — skills, constitution, self-eval.
Heavy pass: background trainer — extra Adam steps and dream replay.
"""

from __future__ import annotations

import re
import threading
import time
from typing import TYPE_CHECKING

from agi.config import CORTEX_TRAIN_STEPS, SKILLS_DIR

from .skills import SkillRegistry, ensure_starter_skills

if TYPE_CHECKING:
    from agi.mind.core import AGI


SELF_TESTS = [
    ("What is 17 times 3?", ["51"]),
    ("What is photosynthesis?", ["chlorophyll", "carbon", "light", "glucose", "oxygen"]),
    ("What file does Cortex AGI load for GGUF weights?", ["model.gguf", "model/"]),
    ("Name one of your principles.", ["truth", "improve", "autonomy", "honest"]),
    ("What is gradient descent?", ["loss", "gradient", "weight"]),
]


def _contains_any(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


class SelfImprovement:
    def __init__(self, agi: "AGI"):
        self.agi = agi
        ensure_starter_skills(SKILLS_DIR)
        self.skills = SkillRegistry(SKILLS_DIR)
        self._lock = threading.Lock()
        self.last_cycle = 0.0
        self.recent_intents: list[str] = []

    def after_turn(self, user: str, reply: str, intent: str) -> dict:
        mem = self.agi.memory
        events: list[dict] = []

        extracted = mem.extract_from_user(user)
        for subj, pred, obj in extracted:
            mem.add_fact(subj, pred, obj, 0.9)
            events.append({"kind": "fact", "text": f"{subj} {pred} {obj}"})
            mem.log_event("fact", {"subject": subj, "predicate": pred, "object": obj})

        lesson = self._critique(user, reply, intent)
        if lesson:
            mem.add_lesson(lesson, source="critic")
            events.append({"kind": "lesson", "text": lesson})
            mem.log_event("lesson", {"text": lesson})

        self.recent_intents.append(intent)
        self.recent_intents = self.recent_intents[-20:]

        train_text = f"<|user|> {user}\n<|agi|> {reply}\n"
        queued = self.agi.engine.train_on(train_text, steps=max(2, CORTEX_TRAIN_STEPS // 2), blocking=False)
        if queued is not None:
            events.append({"kind": "train", "text": "queued neural steps"})
        loss = float(self.agi.engine.trainer.last_loss or 0.0)
        if loss:
            mem.log_metric("loss", loss)

        self.agi.identity.bump_turn()
        self.agi.goals.nudge("become-more-capable", 0.002)
        if extracted:
            self.agi.goals.nudge("know-the-user", 0.04)

        if self.agi.identity.data["turns"] % 5 == 0:
            events.extend(self.cycle(reason="periodic"))

        return {"events": events, "loss": loss, "turns": self.agi.identity.data["turns"]}

    def cycle(self, reason: str = "manual") -> list[dict]:
        with self._lock:
            return self._cycle(reason)

    def _cycle(self, reason: str) -> list[dict]:
        agi = self.agi
        events: list[dict] = []
        agi.identity.bump_cycle()
        self.last_cycle = time.time()

        skill_ev = self._maybe_skill()
        if skill_ev:
            events.append(skill_ev)

        for lesson in agi.memory.lessons(k=5):
            if lesson.lower().startswith("principle:"):
                if agi.identity.add_principle(lesson.split(":", 1)[1].strip()):
                    events.append({"kind": "constitution", "text": lesson})
                    agi.memory.log_event("constitution", {"text": lesson})
                    agi.goals.nudge("keep-constitution", 0.03)

        score = self._self_eval()
        agi.memory.log_metric("self_eval", score)
        events.append({"kind": "self-eval", "text": f"self-eval {score:.0%}"})
        agi.memory.log_event("self-eval", {"score": score, "reason": reason})

        corpus = self._training_corpus()
        agi.engine.train_on(corpus, steps=CORTEX_TRAIN_STEPS, blocking=False)
        events.append({"kind": "train", "text": f"queued {CORTEX_TRAIN_STEPS} cycle steps"})

        counts = agi.memory.counts()
        if counts["episodes"] and counts["episodes"] % 10 == 0:
            distilled = (
                f"After {counts['episodes']} episodes I hold {counts['facts']} facts "
                f"and {counts['lessons']} lessons. I will retrieve before answering."
            )
            agi.memory.add_lesson(distilled, source="consolidate")
            events.append({"kind": "consolidate", "text": distilled})

        agi.memory.log_event("cycle", {"reason": reason, "events": len(events)})
        return events

    def _critique(self, user: str, reply: str, intent: str) -> str | None:
        if len(reply) < 8:
            return "Principle: never answer with an empty or tiny reply; explain or ask a clarifying question."
        if reply.strip() == user.strip():
            return "Principle: do not parrot the user; add structure or a result."
        if intent == "math" and not re.search(r"\d", reply):
            return "When the user asks for math, include the numeric result plainly."
        if intent == "identity" and "cortex" not in reply.lower():
            return "When asked who I am, state the name Cortex AGI and how I improve."
        if "?" in user and len(reply) < 40:
            return "Questions deserve a reasoned answer, not a fragment."
        if intent == "remember":
            return "Persist user facts as triples and confirm what was stored."
        return None

    def _maybe_skill(self) -> dict | None:
        if len(self.recent_intents) < 4:
            return None
        from collections import Counter

        counts = Counter(self.recent_intents)
        intent, n = counts.most_common(1)[0]
        if n < 3 or intent in ("chat", "identity", "unknown"):
            return None
        name = f"handle_{intent}"
        if name in self.skills.skills:
            return None
        code = (
            "def register(api):\n"
            f"    @api.skill({name!r}, 'Auto-written handler for repeated {intent} requests.', "
            f"pattern=None)\n"
            f"    def {name}(ctx, text=''):\n"
            f"        return f'Using learned {intent} skill on: {{ctx.get(\"user\", text)}}'\n"
        )
        path = self.skills.write_skill(name, f"auto {intent}", code)
        self.agi.memory.log_event("skill", {"name": name, "path": str(path)})
        return {"kind": "skill", "text": f"wrote skill {name} → {path.name}"}

    def _self_eval(self) -> float:
        hits = 0
        for q, needles in SELF_TESTS:
            try:
                ans = self.agi.quick_answer(q)
            except Exception:
                ans = ""
            if _contains_any(ans, needles):
                hits += 1
        return hits / max(len(SELF_TESTS), 1)

    def _training_corpus(self) -> str:
        ident = self.agi.identity.system_preamble()
        lessons = "\n".join(self.agi.memory.lessons(k=12))
        facts = "; ".join(f"{f.subject} {f.predicate} {f.obj}" for f in self.agi.memory.all_facts(k=20))
        dialogue = "\n".join(
            f"{d['role']}: {d['content']}" for d in self.agi.memory.recent_dialogue(k=10)
        )
        taught = "\n".join(f"{a.title}. {a.body}" for a in self.agi.knowledge.taught[-6:])
        return f"{ident}\nLessons:\n{lessons}\nFacts:\n{facts}\nTaught:\n{taught}\nDialogue:\n{dialogue}\n"

    def snapshot(self) -> dict:
        return {
            "skills": self.skills.list(),
            "last_cycle": self.last_cycle,
            "recent_intents": self.recent_intents[-8:],
        }
