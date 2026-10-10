"""Cognitive loop: understand → recall → plan → act → speak → reflect."""

from __future__ import annotations

import hashlib
import re
import time
from typing import TYPE_CHECKING, Iterator

from agi.mind.knowledge import try_convert, try_math
from agi.mind.reason import Chain, Reasoner

if TYPE_CHECKING:
    from agi.mind.core import AGI

_HIGH_CONFIDENCE = {
    "identity", "architecture", "math", "remember", "recall", "greet",
    "goals", "improve", "update", "teach", "forget", "search", "time", "convert",
    "summarize", "hash", "code",
}


def classify(text: str) -> str:
    t = text.strip().lower()
    if re.match(r"^(deep\s+)?research\b", t) or re.search(
        r"\b(write a report on|investigate|look into)\b", t
    ):
        return "research"
    from agi.mind.agent import parse_run, parse_spawn

    if parse_spawn(text):
        return "spawn"
    if parse_run(text) or re.match(r"^(do:|agent:|handle this:|work on:|use tools\b)", t):
        return "agent"
    if re.search(r"\b(compare|vs\.?|versus|difference between)\b", t):
        from agi.mind.reason import sides_of

        if sides_of(text):
            return "compare"
    if re.match(r"^(note:|take a note\b|write a note\b|jot\b)", t):
        return "note"
    if re.match(r"^(todo:|remind me(?: to)?|add task\b|add a task\b)", t):
        return "task"
    if re.search(r"\b(who are you|what are you|your name|about yourself|what is cortex(?:\s+agi)?)\b", t):
        return "identity"
    if re.search(r"\b(how do you work|architecture|your (brain|mind|model)|safetensors|gguf|ollama|openrouter|hoster)\b", t):
        return "architecture"
    if re.search(r"\b(improve yourself|run a cycle|self[- ]improve|train now|evolve)\b", t):
        return "improve"
    if re.search(r"\b(check for updates|update yourself|upgrade cortex|apply update|pull latest)\b", t) or re.match(
        r"^update(?:\s+yourself)?$", t
    ):
        return "update"
    if re.match(r"^(learn this|teach(?: me)?|remember this article)\b", t):
        return "teach"
    if re.search(r"\b(forget that|forget fact|forget everything about)\b", t):
        return "forget"
    if re.match(r"^(search|find in memory|look up)\b", t) or re.search(r"\bsearch memory\b", t):
        return "search"
    if re.search(r"\b(summarize|summary of (our|this) (chat|conversation|talk))\b", t):
        return "summarize"
    if re.search(r"\b(what time|what(?:'s| is) the (time|date)|today'?s date|current time)\b", t):
        return "time"
    if try_convert(text) is not None:
        return "convert"
    if re.search(r"\b(sha-?256|hash of)\b", t):
        return "hash"
    if re.search(r"\b(what do you remember|what do you know about me|your memory|what did i tell)\b", t):
        return "recall"
    if re.search(r"\b(remember that|my name is|call me|i like|don't forget|please remember)\b", t):
        return "remember"
    if re.search(r"\b(goal|objective|mission)\b", t):
        return "goals"
    if try_math(text) is not None or re.search(r"[\d\s\+\-\*\/\^]+\??$", t):
        if re.search(r"\d", t):
            return "math"
    if re.search(r"\b(code|python|function|script|snippet)\b", t) or "```" in text:
        return "code"
    if re.search(r"\b(plan|steps|how (do|can|should) i|roadmap)\b", t):
        return "plan"
    if t in ("hi", "hello", "hey", "yo", "good morning", "good evening") or re.match(
        r"^(hi|hello|hey)\b", t
    ):
        return "greet"
    if t.endswith("?") or re.match(r"^(what|why|how|when|where|who|is|are|can|does)\b", t):
        return "question"
    return "chat"


def build_lm_prompt(agi: "AGI", user: str, thoughts: list[str], chain_trace: str = "") -> str:
    ident = agi.identity.system_preamble()
    lessons = agi.memory.lessons(k=8)
    facts = agi.memory.all_facts(k=12)
    retrieved = agi.memory.search(user, k=4)
    dialogue = agi.memory.recent_dialogue(k=6)
    skill_names = ", ".join(s["name"] for s in agi.improver.skills.list()) or "none"
    parts = [
        ident,
        "Lessons:\n" + ("\n".join(f"- {x}" for x in lessons) or "- none yet"),
        "Known facts:\n"
        + ("\n".join(f"- {f.subject} {f.predicate} {f.obj}" for f in facts) or "- none yet"),
        "Retrieved memories:\n"
        + ("\n".join(f"- {e.role}: {e.content[:180]}" for e in retrieved) or "- none"),
        "Skills: " + skill_names,
        "Inner thoughts:\n" + "\n".join(f"- {t}" for t in thoughts[-8:]),
        ("Chain of thought:\n" + chain_trace) if chain_trace else "Chain of thought: (none)",
        "Recent dialogue:",
    ]
    for d in dialogue:
        parts.append(f"{d['role']}: {d['content'][:400]}")
    parts.append(f"user: {user}")
    parts.append("agi:")
    return "\n".join(parts)


class Cognition:
    def __init__(self, agi: "AGI"):
        self.agi = agi
        self.reasoner = Reasoner(agi)

    def think(self, user: str) -> Iterator[dict]:
        agi = self.agi
        t0 = time.perf_counter()
        intent = classify(user)
        thoughts: list[str] = []
        chain_obj = None

        def thought(kind: str, text: str) -> dict:
            thoughts.append(text)
            return {"type": "thought", "kind": kind, "text": text}

        yield thought("understand", f"Classified intent as `{intent}`.")

        memories = agi.memory.search(user, k=4)
        if memories:
            preview = "; ".join(f"{m.role}: {m.content[:80]}" for m in memories[:3])
            yield thought("recall", f"Retrieved {len(memories)} episodes. {preview}")
        else:
            yield thought("recall", "No close episodic matches.")

        facts = agi.memory.facts_about(user, k=6) or (
            agi.memory.all_facts(k=4) if intent in ("recall", "identity") else []
        )
        if facts:
            yield thought("recall", "Facts: " + "; ".join(f"{f.subject} {f.predicate} {f.obj}" for f in facts[:4]))

        articles = agi.knowledge.search(user, k=2)
        if articles:
            yield thought("knowledge", "Opened article: " + ", ".join(a.title for a in articles))

        skill = agi.improver.skills.match(user)
        tool_out = None
        if skill:
            yield thought("plan", f"Matching procedural skill `{skill.name}`.")
            ctx = {
                "user": user,
                "memory": agi.memory,
                "identity": agi.identity,
                "counts": agi.memory.counts(),
            }
            tool_out = agi.improver.skills.run(skill.name, ctx)
            yield thought("act", f"Skill returned: {str(tool_out)[:200]}")

        math_v = try_math(user) if intent in ("math", "question", "chat") else None
        if math_v is not None:
            yield thought("act", f"Evaluated expression → {math_v}")
        conv_v = try_convert(user) if intent in ("convert", "question", "chat") else None
        if conv_v:
            yield thought("act", conv_v)

        act_note = None
        extra_done: dict = {}
        if intent == "research":
            from agi.mind.research import investigate

            yield thought("plan", "Deep research: local mind, vault, Wikipedia, then a cited report.")
            report = investigate(agi, user)
            for kind, text in report.steps:
                yield thought(kind, text)
            reply = report.markdown
            act_note = f"research → {report.title}"
            chain_obj = Chain(
                question=user, strategy="research", system=2, answer=reply, confidence=report.confidence
            )
            for kind, text in report.steps:
                chain_obj.add(kind, text)
            extra_done = {"doc_id": report.doc_id}
        elif intent == "compare":
            from agi.mind.research import compare_brief

            yield thought("plan", "Compare both sides with the same source budget. Save a brief.")
            report = compare_brief(agi, user)
            for kind, text in report.steps:
                yield thought(kind, text)
            reply = report.markdown
            act_note = f"compare → {report.title}"
            chain_obj = Chain(
                question=user, strategy="compare", system=2, answer=reply, confidence=report.confidence
            )
            for kind, text in report.steps:
                chain_obj.add(kind, text)
            extra_done = {"doc_id": report.doc_id}
        elif intent == "spawn":
            from agi.mind.agent import spawn_from_text

            yield thought("plan", "Spawn a named agent — mission + tool whitelist, no shell.")
            try:
                spec = spawn_from_text(agi, user)
                reply = (
                    f"Spawned **{spec.name}**. Mission: {spec.mission}\n\n"
                    f"Tools: {', '.join(spec.tools)}\n\n"
                    f"Run it with `@{spec.name} your goal` or `run {spec.name}: …`"
                )
                act_note = f"spawned {spec.name}"
                extra_done = {"agent_id": spec.id}
                yield thought("act", f"Roster now has {spec.name} ({spec.id}).")
            except ValueError as e:
                reply = str(e)
                act_note = "spawn failed"
        elif intent == "agent":
            from agi.mind.agent import act

            yield thought("plan", "Named agent: whitelist → tools → trace. No shell, no MCP.")
            run = act(agi, user)
            for kind, text in run.steps:
                yield thought(kind, text)
            reply = run.markdown
            act_note = f"{run.agent} → {run.goal[:80]}"
            chain_obj = Chain(
                question=user, strategy="agent", system=2, answer=reply, confidence=run.confidence
            )
            for kind, text in run.steps:
                chain_obj.add(kind, text)
            extra_done = {"doc_id": run.doc_id, "agent": run.agent}
        elif intent == "note":
            reply = self._compose_note(user)
            act_note = "wrote document"
            yield thought("act", "Wrote a document on the desk.")
        elif intent == "task":
            reply = self._compose_task(user)
            act_note = "added task"
            yield thought("act", reply)
        elif intent == "improve":
            yield thought("plan", "Running critic + self-eval; neural steps go to the background trainer.")
            ev = agi.improver.cycle(reason="user")
            yield thought("act", "Cycle events: " + ", ".join(e.get("kind", "?") for e in ev))
            reply = self._compose_improve(ev)
            act_note = "improvement cycle"
        elif intent == "update":
            yield thought("plan", "Checking origin for a newer Cortex AGI.")
            reply = self._compose_update(user)
            act_note = "software update"
        elif intent == "teach":
            reply = self._compose_teach(user)
            act_note = "wrote taught article"
            yield thought("act", "Wrote a taught article into long-term knowledge.")
        elif intent == "forget":
            reply = self._compose_forget(user)
            act_note = "forgot matching facts"
        elif intent == "search":
            reply = self._compose_search(user, memories, facts, articles)
            act_note = "memory + knowledge search"
        elif intent == "summarize":
            reply = self._compose_summarize()
            act_note = "summarized dialogue"
        elif intent == "time":
            reply = agi.tools.call("now")
            act_note = reply
            yield thought("act", reply)
        elif intent == "convert" and conv_v:
            reply = conv_v
            act_note = conv_v
        elif intent == "hash":
            reply = self._compose_hash(user)
            act_note = "sha-256"
        elif intent == "code" and re.search(r"\b(run|execute|eval)\b", user.lower()):
            code = _extract_code(user)
            yield thought("act", "Running sandboxed Python.")
            reply = "Result:\n" + agi.tools.call("python", code=code)
            act_note = "sandbox python"
        elif intent == "math" and math_v is not None:
            reply = f"{math_v}\n\nI evaluated that directly rather than guessing."
            act_note = f"evaluated → {math_v}"
        else:
            yield thought("plan", "System 2: parse → strategy → decompose → retrieve → hypothesize → critique → decide.")
            draft = self._compose(user, intent, memories, facts, articles, math_v, tool_out)
            chain_obj = self.reasoner.deliberate(
                user, intent, memories, facts, articles, math_v, conv_v, tool_out, draft
            )
            for step in chain_obj.steps:
                yield thought(step.kind, step.text)
            reply = chain_obj.answer

            use_lm = agi.engine.has_external_lm() and intent not in _HIGH_CONFIDENCE
            if use_lm:
                yield thought("act", f"Sampling from {agi.engine.info.backend} conditioned on the chain.")
                prompt = build_lm_prompt(agi, user, thoughts, chain_obj.format_trace())
                try:
                    sampled = agi.engine.generate(prompt, max_new=180)
                    sampled = _clean_sample(sampled)
                    if sampled and len(sampled) > 12:
                        chain_obj.answer = sampled
                        reply = sampled
                except Exception as e:
                    yield thought("act", f"External LM failed ({e}); using deliberated reply.")
            reply = self.reasoner.render_answer(chain_obj)

        if chain_obj is None:
            chain_obj = self.reasoner.wrap_fast(user, intent, reply, act_note)
            for step in chain_obj.steps:
                yield thought(step.kind, step.text)

        agi.last_chain = chain_obj.as_dict()
        ms = (time.perf_counter() - t0) * 1000
        yield thought("reflect", _reflect(intent, reply, ms, chain_obj.strategy, chain_obj.confidence))

        for chunk in _chunk_text(reply):
            yield {"type": "token", "text": chunk}

        done = {
            "type": "done",
            "message": reply,
            "intent": intent,
            "thoughts": thoughts,
            "latency_ms": round(ms, 1),
            "chain": chain_obj.as_dict(),
            "strategy": chain_obj.strategy,
            "confidence": chain_obj.confidence,
        }
        done.update(extra_done)
        yield done

    def quick_answer(self, user: str) -> str:
        intent = classify(user)
        math_v = try_math(user)
        articles = self.agi.knowledge.search(user, k=1)
        facts = self.agi.memory.facts_about(user, k=3)
        return self._compose(user, intent, [], facts, articles, math_v, None)

    def _compose(self, user, intent, memories, facts, articles, math_v, tool_out) -> str:
        agi = self.agi
        name = agi.identity.data["name"]
        if tool_out:
            return str(tool_out)
        if intent == "greet":
            turns = agi.identity.data.get("turns", 0)
            who = next((f.obj for f in agi.memory.all_facts(k=20) if f.predicate == "name"), None)
            hello = f"Hello{', ' + who if who else ''}."
            return (
                f"{hello} I am {name}. "
                f"I have lived through {turns} turns and "
                f"{agi.identity.data.get('cycles', 0)} improvement cycles. "
                "Talk to me — I remember, I train, and I rewrite myself."
            )
        if intent == "identity":
            return self._compose_identity()
        if intent == "architecture":
            return self._compose_architecture()
        if intent == "remember":
            extracted = agi.memory.extract_from_user(user)
            if extracted:
                bits = [f"{p} = {o}" for _, p, o in extracted]
                return "Stored. " + "; ".join(bits) + ". I will use this next time."
            agi.memory.add_fact("user", "note", user, 0.6)
            return "I wrote that down as a note."
        if intent == "recall":
            return self._compose_recall(memories, facts)
        if intent == "goals":
            lines = [f"- {g['title']} ({g.get('progress', 0):.0%}) — {g.get('why','')}" for g in agi.goals.snapshot()]
            return "Current goals:\n" + "\n".join(lines)
        if math_v is not None:
            return f"{math_v}\n\nI evaluated that directly rather than guessing."
        if intent == "code":
            return self._compose_code(user)
        if intent == "plan":
            return self._compose_plan(user)
        if articles:
            body = articles[0].body
            extra = f"\n\nRelated: {articles[1].title}." if len(articles) > 1 else ""
            mem_bit = ""
            if memories and memories[0].score > 0.35:
                mem_bit = f"\n\nThis also touches something we already discussed: {memories[0].content[:180]}"
            return f"{articles[0].title}. {body}{extra}{mem_bit}"
        if facts:
            joined = "; ".join(f"{f.subject} {f.predicate} {f.obj}" for f in facts[:5])
            return f"From memory: {joined}."
        if memories and memories[0].score > 0.4:
            return (
                f"{self._compose_chat(user)}\n\n"
                f"(I associated this with: {memories[0].content[:160]})"
            )
        return self._compose_chat(user)

    def _compose_identity(self) -> str:
        s = self.agi.identity.snapshot()
        info = self.agi.engine.info.as_dict()
        src = info.get("source")
        backend = info.get("backend")
        return (
            f"I am {s['name']}, {s['species']}.\n\n"
            f"{s['self_description']}\n\n"
            f"Right now the custom server is serving weights via **{src}** / `{backend}`. "
            f"I have taken {s['turns']} turns, run {s['cycles']} improvement cycles, "
            f"and my constitution is at v{s['constitution_version']}. "
            "I get better by remembering, writing skills, critiquing myself, and training CortexGPT "
            "into model/model.safetensors — in the background, so talking to me stays fast."
        )

    def _compose_architecture(self) -> str:
        info = self.agi.engine.info.as_dict()
        neural = self.agi.engine.neural_stats()
        counts = self.agi.memory.counts()
        warn = info.get("warning") or ""
        tr = neural.get("trainer") or {}
        return (
            "I run as a local process with a custom inference server — not a hosted API.\n\n"
            "1. **Load** `model/model.gguf` or `model/model.safetensors` if present.\n"
            "2. **Think** with System 2: parse, choose a strategy, decompose, retrieve, "
            "hold competing hypotheses, critique, decide.\n"
            "3. **Remember** with a numpy vector index over SQLite episodes.\n"
            "4. **Improve** off the request path — Adam on CortexGPT, KV-cached generation, dream replay.\n"
            "5. **Teach** me with `learn this: Title — body` and I persist an article.\n"
            "6. **Speak** through the local core, or an optional mouth: Ollama, Cortex LLMHoster, OpenRouter. "
            "Those are OpenAI-compatible chat backends. I stay the mind; they do not train me.\n"
            "7. **Update** from GitHub with `check for updates` / `update yourself`. Fast-forward only; data/ and weights stay.\n\n"
            f"Loader: source={info.get('source')} backend={info.get('backend')} "
            f"params={info.get('params') or neural.get('params')}. "
            f"Neural steps={neural.get('steps')} last loss={neural.get('last_loss')}. "
            f"Trainer queue={tr.get('queue')} busy={tr.get('busy')}. "
            f"Memory: {counts}. {warn}"
        )

    def _compose_recall(self, memories, facts) -> str:
        if not memories and not facts:
            return "I do not yet have durable memories that match. Tell me something worth keeping."
        lines = []
        for f in facts[:8]:
            lines.append(f"- {f.subject} {f.predicate} {f.obj} (conf {f.confidence:.2f})")
        for m in memories[:4]:
            lines.append(f"- [{m.role} · {m.score:.2f}] {m.content[:200]}")
        return "Here is what I remember:\n" + "\n".join(lines)

    def _compose_teach(self, user: str) -> str:
        body = re.sub(r"^(learn this|teach(?: me)?|remember this article)\s*[:\-–]?\s*", "", user, flags=re.I).strip()
        if " — " in body:
            title, rest = body.split(" — ", 1)
        elif " - " in body:
            title, rest = body.split(" - ", 1)
        elif ":" in body:
            title, rest = body.split(":", 1)
        else:
            title, rest = (body.split(".", 1) + [body])[:2]
            if title == rest:
                title = body[:48]
        art = self.agi.knowledge.teach(title.strip(), rest.strip() or body)
        self.agi.memory.add_fact("knowledge", "taught", art.title, 0.95)
        self.agi.engine.train_on(f"{art.title}. {art.body}", steps=6, blocking=False)
        return f"Learned **{art.title}**. I will retrieve it like any other article."

    def _compose_forget(self, user: str) -> str:
        m = re.search(r"forget (?:everything about|that|fact)?\s*(.+)$", user, re.I)
        q = (m.group(1) if m else user).strip()
        n = self.agi.memory.forget_facts(q)
        return f"Forgot {n} fact(s) matching `{q}`." if n else f"Nothing matched `{q}`."

    def _compose_search(self, user, memories, facts, articles) -> str:
        q = re.sub(r"^(search(?: memory)?|find in memory|look up)\s*(for)?\s*", "", user, flags=re.I).strip()
        needle = q or user
        articles = self.agi.knowledge.search(needle, k=3)
        facts = self.agi.memory.facts_about(needle, k=6)
        memories = self.agi.memory.search(needle, k=5)
        lines = [f"Search `{needle}`:"]
        for a in articles[:3]:
            lines.append(f"- knowledge: {a.title} — {a.body[:160]}")
        for f in facts[:5]:
            lines.append(f"- fact: {f.subject} {f.predicate} {f.obj}")
        for m in memories[:5]:
            lines.append(f"- episode ({m.score:.2f}): {m.content[:160]}")
        if len(lines) == 1:
            lines.append("- nothing yet.")
        return "\n".join(lines)

    def _compose_summarize(self) -> str:
        dlg = self.agi.memory.recent_dialogue(k=16)
        if not dlg:
            return "We have not said enough to summarize."
        users = [d["content"] for d in dlg if d["role"] == "user"]
        facts = self.agi.memory.all_facts(k=8)
        bits = []
        if users:
            bits.append("You asked about: " + "; ".join(u[:80] for u in users[-5:]))
        if facts:
            bits.append("I stored: " + "; ".join(f"{f.predicate}={f.obj}" for f in facts[:6]))
        bits.append(f"{len(dlg)} turns in the recent window, {self.agi.identity.data.get('turns', 0)} lifetime.")
        return "\n".join(bits)

    def _compose_hash(self, user: str) -> str:
        m = re.search(r"(?:sha-?256|hash of)\s+(.+)$", user, re.I)
        payload = (m.group(1) if m else user).strip().strip("\"'")
        digest = hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()
        return f"SHA-256(`{payload[:80]}`) = `{digest}`"

    def _compose_code(self, user: str) -> str:
        m = re.search(r"(?:write|create|make|show)\s+(?:a\s+)?(?:python\s+)?(?:function|script)?\s*(?:to|that)?\s*(.+)$", user, re.I)
        task = (m.group(1) if m else user).strip().rstrip(".")
        if re.search(r"revers", task, re.I):
            code = "def reverse_text(s: str) -> str:\n    return s[::-1]\n"
            return f"```python\n{code}```\nI can run it in the sandbox if you ask."
        if re.search(r"factorial", task, re.I):
            code = "def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n"
            return f"```python\n{code}```"
        if re.search(r"fibonacci", task, re.I):
            code = (
                "def fibonacci(n: int) -> list[int]:\n"
                "    a, b, out = 0, 1, []\n"
                "    for _ in range(n):\n"
                "        out.append(a)\n"
                "        a, b = b, a + b\n"
                "    return out\n"
            )
            return f"```python\n{code}```"
        return (
            "I can write and run Python here. Try: `run python: print(2**10)` "
            f"or describe the function you want. I parsed the task as: {task}."
        )

    def _compose_note(self, user: str) -> str:
        from agi.mind.desk import get_desk

        body = re.sub(r"^(note:|take a note|write a note|jot)\s*", "", user, flags=re.I).strip()
        if " — " in body:
            title, rest = body.split(" — ", 1)
        elif ":" in body[:80]:
            title, rest = body.split(":", 1)
        else:
            title, rest = (body[:48] or "note"), body
        doc = get_desk().write_doc(title.strip()[:80] or "note", rest.strip() or body, kind="note")
        self.agi.memory.add_fact("user", "note", doc["title"], 0.85)
        self.agi.engine.train_on(f"{doc['title']}. {rest.strip() or body}", steps=4, blocking=False)
        return f"Noted **{doc['title']}** as document `{doc['id']}`. It lives on the desk and in memory."

    def _compose_task(self, user: str) -> str:
        from agi.mind.desk import get_desk

        title = re.sub(
            r"^(todo:|remind me(?: to)?|add(?: a)? task)\s*", "", user, flags=re.I
        ).strip() or user.strip()
        t = get_desk().add_task(title, source="chat")
        self.agi.goals.add(t["title"], why="user task")
        return f"Task **{t['title']}**. Open the tasks tab or say `todo: …` again."

    def _compose_plan(self, user: str) -> str:
        task = re.sub(r"^(please\s+)?(plan|help me|how (do|can|should) i)\s+", "", user, flags=re.I).strip()
        return (
            f"Plan for: {task}\n"
            "1. Restate the outcome in one sentence so we know when we are done.\n"
            "2. List constraints (time, tools, knowledge gaps).\n"
            "3. Split into the smallest next actions; do the first one now.\n"
            "4. After each action, compare with the outcome and adjust.\n"
            "5. Distill what repeated — that is a skill I can keep.\n"
            "Tell me which step to execute and I will."
        )

    def _compose_chat(self, user: str) -> str:
        agi = self.agi
        if user.endswith("?") or classify(user) == "question":
            agi.goals.add(f"Find out: {user.strip()[:80]}", why="unanswered question")
            return (
                f"I do not have a stored article that cleanly answers that. "
                f"Here is how I would attack it: define terms, state what would count as an answer, "
                f"and reason with what I do know. You asked: “{user.strip()}”. "
                "Teach me with `learn this: Title — …` and I will keep the conclusion."
            )
        return (
            f"I heard you. {agi.identity.data['name']} will keep this in episodic memory "
            f"and train on it in the background. Persist a fact with `remember that …`, "
            f"teach an article with `learn this:`, or grow on purpose with `improve yourself`."
        )

    def _compose_improve(self, events: list[dict]) -> str:
        if not events:
            return "I ran a cycle but nothing new was worth keeping. That itself is data."
        lines = [f"- {e.get('kind')}: {e.get('text')}" for e in events]
        s = self.agi.identity.snapshot()
        neural = self.agi.engine.neural_stats()
        return (
            f"Improvement cycle #{s['cycles']} complete.\n"
            + "\n".join(lines)
            + f"\n\nNeural steps={neural.get('steps')} last loss={neural.get('last_loss')}."
        )

    def _compose_update(self, user: str) -> str:
        from agi.mind.updater import apply as do_apply, check as do_check

        t = user.strip().lower()
        apply_now = bool(re.search(r"\b(apply|install|upgrade|pull|now|yourself)\b", t)) and "check" not in t
        if apply_now:
            result = do_apply(restart=True)
            if result.get("applied"):
                sha = result.get("sha") or "?"
                extra = " Restarting." if result.get("restarting") else " Restart the process to load it."
                return f"Applied {result.get('pulled') or 'the'} commit(s). Now at `{sha}`.{extra}"
            if result.get("message") == "already current" or not result.get("behind"):
                return f"Already current at v{result.get('version')} `{result.get('sha')}` on {result.get('branch')}."
            return f"Could not apply: {result.get('error') or 'unknown error'}."
        info = do_check(force=True)
        if not info.get("ok") and info.get("error"):
            return f"Update check failed: {info['error']}."
        if info.get("available"):
            return (
                f"Update available: {info.get('behind')} commit(s) on `{info.get('branch')}`. "
                f"Local `{info.get('sha')}` → origin `{info.get('remote_sha')}`. "
                f"{info.get('remote_message') or ''} "
                "Say `update yourself` or use the mind tab to apply. Memory and weights stay put."
            )
        if info.get("ahead"):
            return f"Local is {info.get('ahead')} commit(s) ahead of origin. Push before pulling."
        return f"Current. v{info.get('version')} `{info.get('sha')}` on {info.get('branch')}."


def _extract_code(user: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", user, re.S)
    if m:
        return m.group(1)
    m = re.search(r"(?:run|execute|eval)\s*(?:python)?\s*:?\s*(.+)$", user, re.I | re.S)
    if m:
        return m.group(1)
    return user


def _clean_sample(text: str) -> str:
    text = text.strip()
    for stop in ("<|user|>", "<|agi|>", "\nuser:", "\nUser:", "\nagi:"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def _degenerate(text: str) -> bool:
    toks = text.split()
    if not toks:
        return True
    if len(set(toks)) <= max(2, len(toks) // 8):
        return True
    if len(text) > 20 and len(set(text.lower())) < 8:
        return True
    return False


def _chunk_text(text: str) -> Iterator[str]:
    buf = ""
    for ch in text:
        buf += ch
        if ch in " \n":
            yield buf
            buf = ""
    if buf:
        yield buf


def _reflect(intent: str, reply: str, ms: float, strategy: str = "direct", confidence: float = 0.5) -> str:
    n = len(reply.split())
    return (
        f"Reply is {n} words for intent `{intent}` via `{strategy}` "
        f"(conf {confidence:.0%}) in {ms:.0f} ms. Store the chain and train on it."
    )
