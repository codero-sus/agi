"""Chain-of-thought / System-2 deliberation.

A general intelligence is not a single lookup. It restates the question,
picks a strategy, splits the work, gathers evidence, holds competing
hypotheses, attacks its own answer, then speaks. This module is that loop.

Fast intents (math, time, convert) still get a short chain so every reply
has a trace. Hard intents run the full deliberative path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agi.mind.knowledge import try_convert, try_math

if TYPE_CHECKING:
    from agi.mind.core import AGI
    from agi.mind.knowledge import Article
    from agi.mind.memory import Episode, Fact


FAST_INTENTS = {
    "greet", "remember", "forget", "teach", "search", "time",
    "convert", "hash", "improve", "summarize", "math",
}


@dataclass
class Step:
    kind: str
    text: str
    confidence: float = 0.7
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:4],
        }


@dataclass
class Hypothesis:
    label: str
    answer: str
    score: float
    why: str


@dataclass
class Chain:
    question: str
    strategy: str
    steps: list[Step] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.5
    system: int = 2  # 1 = fast, 2 = deliberative

    def add(self, kind: str, text: str, confidence: float = 0.7, evidence: list[str] | None = None) -> Step:
        step = Step(kind, text, confidence, evidence or [])
        self.steps.append(step)
        return step

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "strategy": self.strategy,
            "system": self.system,
            "confidence": round(self.confidence, 3),
            "steps": [s.as_dict() for s in self.steps],
            "hypotheses": [
                {"label": h.label, "score": round(h.score, 3), "why": h.why}
                for h in self.hypotheses
            ],
        }

    def format_trace(self) -> str:
        lines = [f"**Thinking** · {self.strategy} · system-{self.system} · conf {self.confidence:.0%}"]
        for i, s in enumerate(self.steps, 1):
            lines.append(f"{i}. **{s.kind}** — {s.text}")
        return "\n".join(lines)


def question_kind(text: str) -> str:
    t = text.strip().lower()
    if re.match(r"^why\b", t) or re.search(r"\bwhy (does|do|is|are|would|did)\b", t):
        return "why"
    if re.match(r"^how\b", t) or re.search(r"\bhow (does|do|can|should|to|would|did)\b", t):
        return "how"
    if re.search(r"\b(what if|suppose|imagine if|counterfactual)\b", t):
        return "whatif"
    if re.search(r"\b(vs\.?|versus|compare|compared to|difference between|or rather)\b", t):
        return "compare"
    if re.search(r"\b(should i|ought|ethic|right thing|trade[- ]?off)\b", t):
        return "normative"
    if re.match(r"^what\b", t):
        return "what"
    if re.match(r"^(who|when|where)\b", t):
        return "slot"
    if re.search(r"\b(plan|steps|roadmap|how (do|should) i)\b", t):
        return "plan"
    return "other"


def select_strategy(text: str, intent: str, qkind: str, has_knowledge: bool) -> str:
    if intent in ("math", "convert", "hash") or try_math(text) or try_convert(text):
        return "compute"
    if qkind == "why":
        return "causal"
    if qkind == "how":
        return "mechanism"
    if qkind == "compare":
        return "compare"
    if qkind == "whatif":
        return "counterfactual"
    if qkind == "plan" or intent == "plan":
        return "means-ends"
    if qkind == "normative":
        return "principled"
    if has_knowledge:
        return "retrieve"
    if intent in ("question", "chat"):
        return "first-principles"
    return "direct"


def _strip_prompt(text: str) -> str:
    return re.sub(
        r"^(please\s+)?(tell me |explain |describe |what(?:'s| is) |why |how )",
        "",
        text.strip().rstrip("?"),
        flags=re.I,
    ).strip()


def sides_of(text: str) -> tuple[str, str] | None:
    return _pair(text)


def _pair(text: str) -> tuple[str, str] | None:
    t = text.strip().rstrip("?")
    m = re.search(r"difference between (.+) and (.+)$", t, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"compare (.+) (?:and|with|to) (.+)$", t, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)$", t, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _topic(text: str, qkind: str) -> str:
    t = text.strip().rstrip("?")
    m = re.match(
        r"^why (?:does|do|did)\s+(.+?)\s+(produce|cause|create|lead to|result in)\s+(.+)$",
        t,
        re.I,
    )
    if m:
        return m.group(1).strip()
    t = re.sub(r"^what if\s+", "", t, flags=re.I)
    patterns = [
        r"^why (?:does|do|did|is|are|would|can|can't)\s+",
        r"^how (?:does|do|did|can|should|to|would|is|are)\s+",
        r"^what (?:is|are|was|were|does|do)\s+(?:the\s+)?",
        r"^who (?:is|are|was|were)\s+",
    ]
    for p in patterns:
        t2 = re.sub(p, "", t, flags=re.I)
        if t2 != t:
            return t2.strip()
    return _strip_prompt(t)


class Reasoner:
    def __init__(self, agi: "AGI"):
        self.agi = agi

    def wrap_fast(self, user: str, intent: str, answer: str, act: str | None = None) -> Chain:
        chain = Chain(question=user, strategy="direct", system=1, confidence=0.9, answer=answer)
        chain.add("parse", f"Intent `{intent}` is high-confidence — System 1 is enough.", 0.95)
        if act:
            chain.add("act", act, 0.95)
        chain.add("decide", "Return the computed or stored result without extra search.", 0.95)
        return chain

    def deliberate(
        self,
        user: str,
        intent: str,
        memories: list,
        facts: list,
        articles: list,
        math_v: str | None,
        conv_v: str | None,
        tool_out: str | None,
        draft: str,
    ) -> Chain:
        qkind = question_kind(user)
        has_k = bool(articles)
        strategy = select_strategy(user, intent, qkind, has_k)
        chain = Chain(question=user, strategy=strategy, system=2, answer=draft)
        topic = _topic(user, qkind)

        chain.add(
            "parse",
            f"Restate: “{topic or user.strip()}”. Kind=`{qkind}`. "
            f"A good answer would be specific, sourced, and honest about gaps.",
            0.8,
        )
        chain.add(
            "strategy",
            f"Select `{strategy}` "
            + {
                "causal": "— look for mechanism, not a slogan.",
                "mechanism": "— list parts, then the sequence they fire in.",
                "compare": "— characterize each side, then the joint and the split.",
                "counterfactual": "— hold the world fixed except one change, then follow consequences.",
                "means-ends": "— goal, gap, next action.",
                "principled": "— apply the constitution, then the tradeoff.",
                "retrieve": "— ground in stored articles and memory, then synthesize.",
                "first-principles": "— define terms, name constraints, deduce, mark uncertainty.",
                "compute": "— evaluate; do not guess a number.",
                "direct": "— one hop.",
            }.get(strategy, "."),
            0.85,
        )

        subs = self._decompose(user, qkind, topic, articles, strategy)
        chain.add(
            "decompose",
            "Subquestions: " + "; ".join(f"({i+1}) {s}" for i, s in enumerate(subs)),
            0.75,
        )

        findings: list[str] = []
        for sub in subs:
            hit = self._answer_sub(sub, articles, memories, facts)
            findings.append(hit["claim"])
            chain.add(
                "retrieve" if hit["source"] != "first-principles" else "deduce",
                f"{sub} → {hit['claim']}",
                hit["confidence"],
                hit["evidence"],
            )

        if math_v:
            chain.add("compute", f"Direct evaluation = {math_v}", 0.99, [math_v])
            findings.append(f"numeric result {math_v}")
        if conv_v:
            chain.add("compute", conv_v, 0.99, [conv_v])
        if tool_out:
            chain.add("act", f"Skill/tool: {str(tool_out)[:220]}", 0.8)

        hyps = self._hypotheses(user, intent, strategy, qkind, topic, articles, findings, draft)
        chain.hypotheses = hyps
        if hyps:
            ranked = ", ".join(f"{h.label} {h.score:.0%}" for h in hyps)
            chain.add("hypothesize", f"Candidates: {ranked}.", max(h.score for h in hyps))

        best = hyps[0] if hyps else Hypothesis("draft", draft, 0.55, "composer draft")
        critique = self._critique(user, best, articles, strategy)
        chain.add("critique", critique["text"], critique["confidence"])

        final, conf = self._decide(user, intent, strategy, qkind, topic, best, findings, articles, critique, draft)
        chain.answer = final
        chain.confidence = conf
        chain.add(
            "decide",
            f"Commit to `{best.label}` after critique. Confidence {conf:.0%}.",
            conf,
        )
        return chain

    def _decompose(self, user: str, qkind: str, topic: str, articles: list, strategy: str) -> list[str]:
        pair = _pair(user)
        if strategy == "compare" and pair:
            a, b = pair
            return [
                f"What is {a}?",
                f"What is {b}?",
                f"What do {a} and {b} share?",
                f"Where do {a} and {b} conflict?",
            ]
        if strategy == "causal":
            return [
                f"What is {topic}?",
                f"What mechanism produces {topic}?",
                f"What would fail if that mechanism stopped?",
            ]
        if strategy == "mechanism":
            return [
                f"What are the parts of {topic}?",
                f"In what order do they act in {topic}?",
                f"What is the output of {topic}?",
            ]
        if strategy == "counterfactual":
            return [
                f"What is the actual world regarding {topic}?",
                f"What single change is being imagined?",
                f"Which consequences follow and which do not?",
            ]
        if strategy == "means-ends":
            return [
                f"What does done look like for: {topic}?",
                "What is the smallest next action?",
                "How will we know the action worked?",
            ]
        if strategy == "principled":
            return [
                "Which constitution clauses apply?",
                "What are the competing goods?",
                "What reversible action preserves option value?",
            ]
        subs = []
        for art in articles[:3]:
            subs.append(f"What does “{art.title}” contribute?")
        if not subs:
            subs = [f"What do I already know about {topic or user}?", "What would count as evidence?"]
        return subs[:4]

    def _answer_sub(self, sub: str, articles: list, memories: list, facts: list) -> dict:
        agi = self.agi
        found = agi.knowledge.search(sub, k=2)
        if found:
            art = found[0]
            return {
                "claim": f"{art.title}: {art.body[:280]}",
                "confidence": 0.82,
                "evidence": [art.title],
                "source": "knowledge",
            }
        if articles:
            art = articles[0]
            return {
                "claim": f"From {art.title}: {art.body[:240]}",
                "confidence": 0.7,
                "evidence": [art.title],
                "source": "knowledge",
            }
        if facts:
            f = facts[0]
            return {
                "claim": f"Memory triple {f.subject} {f.predicate} {f.obj}",
                "confidence": float(getattr(f, "confidence", 0.6)),
                "evidence": [f"{f.predicate}:{f.obj}"],
                "source": "memory",
            }
        if memories:
            m = memories[0]
            return {
                "claim": f"Prior episode: {m.content[:180]}",
                "confidence": min(0.55, 0.3 + float(getattr(m, "score", 0))),
                "evidence": ["episode"],
                "source": "memory",
            }
        return {
            "claim": f"No stored article for “{sub}”. Reason from definitions and constraints.",
            "confidence": 0.4,
            "evidence": [],
            "source": "first-principles",
        }

    def _hypotheses(
        self,
        user: str,
        intent: str,
        strategy: str,
        qkind: str,
        topic: str,
        articles: list,
        findings: list[str],
        draft: str,
    ) -> list[Hypothesis]:
        hyps: list[Hypothesis] = []
        grounded = self._synthesize(strategy, qkind, topic, articles, findings, user)
        hyps.append(Hypothesis("grounded", grounded, 0.78 if articles else 0.5, "stitched from retrieval"))
        fp = self._first_principles(topic or user, qkind, strategy)
        hyps.append(Hypothesis("first-principles", fp, 0.55, "definitions → constraints → implication"))
        if articles and len(articles) > 1:
            analog = (
                f"Treat this like {articles[0].title}, which already lives in memory. "
                f"{articles[0].body[:200]} Analogous structure may apply to {topic}."
            )
            hyps.append(Hypothesis("analogical", analog, 0.58, f"map onto {articles[0].title}"))
        if draft and draft not in (grounded, fp):
            prior = 0.9 if intent in ("identity", "architecture", "greet", "recall", "goals", "code") else 0.52
            hyps.append(Hypothesis("composer", draft, prior, "specialist composer"))
        hyps.sort(key=lambda h: h.score, reverse=True)
        return hyps[:3]

    def _synthesize(self, strategy: str, qkind: str, topic: str, articles: list, findings: list[str], user: str) -> str:
        if strategy == "compare":
            pair = _pair(user)
            if pair:
                a, b = pair
                aa = self.agi.knowledge.search(a, k=1)
                bb = self.agi.knowledge.search(b, k=1)
                left = aa[0].body[:280] if aa else f"(no stored article for {a})"
                right = bb[0].body[:280] if bb else f"(no stored article for {b})"
                return (
                    f"**{a}** — {left}\n\n"
                    f"**{b}** — {right}\n\n"
                    "Shared: both are maps of structure. "
                    "Split: they use different primitives and fail on each other's turf. "
                    "A general intelligence keeps both maps and knows which terrain each one covers."
                )
        if strategy == "causal" and articles:
            art = articles[0]
            return (
                f"{topic.capitalize() if topic else art.title} happens because of a mechanism, not a label. "
                f"{art.body} "
                "If that mechanism stopped, the effect would stop with it — that is the causal test."
            )
        if strategy == "mechanism" and articles:
            art = articles[0]
            return (
                f"Parts and sequence for {topic or art.title}: {art.body} "
                "Read it as a pipeline: inputs → transformations → outputs. Skip a stage and the output vanishes."
            )
        if strategy == "counterfactual":
            base = articles[0].body if articles else "the actual world as I know it"
            return (
                f"Hold everything fixed except the imagined change. Baseline: {base[:240]} "
                f"The ‘what if’ only moves consequences that actually depend on that change — "
                "not the entire universe."
            )
        if strategy == "means-ends":
            return (
                f"Goal: {topic}. Gap: I do not yet have a finished artifact. "
                "Next action: write the outcome in one sentence, then do the smallest step that "
                "would make that sentence truer. Repeat. Distill whatever repeats into a skill."
            )
        if strategy == "principled":
            pr = self.agi.identity.data.get("principles") or []
            clause = pr[0] if pr else "Seek truth. When uncertain, say so."
            return (
                f"Governing clause: {clause} "
                f"Applied to “{topic}”: prefer the reversible move, name the tradeoff, do not pretend certainty."
            )
        if articles:
            body = articles[0].body
            extra = f"\n\nAlso relevant: {articles[1].title}." if len(articles) > 1 else ""
            joined = " ".join(findings[:2])
            return f"{articles[0].title}. {body}{extra}\n\n{joined}".strip()
        if findings:
            return findings[0]
        return f"I do not yet have a grounded account of {topic or user}."

    def _first_principles(self, topic: str, qkind: str, strategy: str) -> str:
        return (
            f"First principles on “{topic}”. "
            "1. Name the thing in simpler words. "
            "2. List what must be true for it to exist (constraints). "
            "3. Drop anything that is not forced by those constraints. "
            "4. Say what would falsify the remaining claim. "
            f"Question kind `{qkind}` / strategy `{strategy}`: I will not invent citations I do not have."
        )

    def _critique(self, user: str, best: Hypothesis, articles: list, strategy: str) -> dict:
        holes = []
        if best.score < 0.6:
            holes.append("leading hypothesis is weakly evidenced")
        if not articles and strategy in ("causal", "mechanism", "retrieve"):
            holes.append("no stored article — causal claims may be stories")
        if len(best.answer) < 40:
            holes.append("answer is too thin for the question")
        # Did we answer a different question?
        topic_toks = [t for t in re.split(r"\W+", user.lower()) if len(t) > 3]
        if topic_toks:
            covered = sum(1 for t in topic_toks if t in best.answer.lower())
            if covered / len(topic_toks) < 0.2:
                holes.append("possible question substitution — few topic tokens appear in the answer")
        if strategy == "causal" and not re.search(r"\b(because|cause|mechanism|produces|leads)\b", best.answer, re.I):
            holes.append("causal strategy produced no causal language — rewrite toward mechanism")
        if not holes:
            return {
                "text": "Devil's advocate finds no fatal hole. Residual risk: unknown unknowns, not contradiction.",
                "confidence": 0.8,
                "holes": [],
            }
        return {
            "text": "Devil's advocate: " + "; ".join(holes) + ". I will hedge and invite correction.",
            "confidence": 0.55,
            "holes": holes,
        }

    def _decide(
        self,
        user: str,
        intent: str,
        strategy: str,
        qkind: str,
        topic: str,
        best: Hypothesis,
        findings: list[str],
        articles: list,
        critique: dict,
        draft: str,
    ) -> tuple[str, float]:
        conf = best.score
        if critique.get("holes"):
            conf = max(0.28, conf - 0.12 * min(3, len(critique["holes"])))
        if articles:
            conf = min(0.93, conf + 0.08)

        body = best.answer
        if critique.get("holes") and "hedge" in critique["text"]:
            body = body.rstrip() + "\n\nI am not fully sure — teach me a better account with `learn this:` if this is wrong."
        return body, conf

    def render_answer(self, chain: Chain) -> str:
        """User-facing answer with the chain of thinking underneath."""
        body = (chain.answer or "").rstrip()
        if chain.system == 1:
            return body
        lines = [
            body,
            "",
            chain.format_trace(),
        ]
        return "\n".join(lines).strip()
