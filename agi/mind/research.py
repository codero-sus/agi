"""Deep research that writes a cited report into the mind.

Odysseus wraps a research agent around someone else's model.
CORTEX researches with System 2, then keeps the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agi.mind.desk import get_desk
from agi.mind.net import wiki_search, wiki_summary
from agi.mind.reason import Chain, sides_of
from agi.mind.workspace import get_workspace

if TYPE_CHECKING:
    from agi.mind.core import AGI


def topic_of(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^(please\s+)?(deep\s+)?research(?:\s+this)?(?:\s+on)?:\s*", "", t, flags=re.I)
    t = re.sub(r"^(write a report on|investigate|look into)\s+", "", t, flags=re.I)
    t = t.strip().rstrip("?")
    return t[:160] or text.strip()[:160]


@dataclass
class Source:
    kind: str
    title: str
    excerpt: str
    url: str = ""
    score: float = 0.0


@dataclass
class Report:
    topic: str
    title: str
    markdown: str
    sources: list[Source] = field(default_factory=list)
    steps: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.5
    doc_id: str | None = None
    chain: dict | None = None
    tasks: list[dict] = field(default_factory=list)


def investigate(agi: "AGI", user: str) -> Report:
    topic = topic_of(user)
    chain = Chain(question=topic, strategy="research", system=2, confidence=0.55)
    steps: list[tuple[str, str]] = []

    def add(kind: str, text: str, conf: float = 0.7, evidence: list[str] | None = None) -> None:
        chain.add(kind, text, conf, evidence)
        steps.append((kind, text))

    add("parse", f"Research question: {topic}. A report needs a verdict, evidence, gaps, and next actions.")
    add("strategy", "Local knowledge + vault + memory, then Wikipedia. Synthesize; do not paste.")

    sources: list[Source] = []

    articles = agi.knowledge.search(topic, k=4)
    if articles:
        add("retrieve", "Knowledge: " + ", ".join(a.title for a in articles), 0.8, [a.title for a in articles])
        for a in articles:
            sources.append(Source("knowledge", a.title, a.body[:900], score=0.85))
    else:
        add("retrieve", "No built-in article matched closely.")

    facts = agi.memory.facts_about(topic, k=6)
    if facts:
        blob = "; ".join(f"{f.subject} {f.predicate} {f.obj}" for f in facts[:6])
        add("retrieve", f"Memory facts: {blob[:240]}", 0.7)
        sources.append(Source("memory", "semantic facts", blob[:900], score=0.6))

    episodes = agi.memory.search(topic, k=3)
    if episodes:
        add("retrieve", f"Episodes: {len(episodes)} nearby traces.")
        sources.append(
            Source("memory", "episodes", " | ".join(e.content[:180] for e in episodes), score=0.5)
        )

    vault_hits = get_workspace().search_vault(topic, k=4)
    if vault_hits:
        add("retrieve", "Vault: " + ", ".join(v["name"] for v in vault_hits), 0.75, [v["name"] for v in vault_hits])
        for v in vault_hits:
            sources.append(Source("vault", v["name"], v.get("excerpt") or "", score=0.8))

    add("retrieve", "Querying Wikipedia for independent coverage.")
    wiki_hits = wiki_search(topic, k=4)
    if wiki_hits:
        add("retrieve", "Wikipedia hits: " + ", ".join(h["title"] for h in wiki_hits[:4]), 0.8)
        for hit in wiki_hits[:3]:
            summ = wiki_summary(hit["title"])
            if summ:
                sources.append(
                    Source("wikipedia", summ["title"], summ["extract"], url=summ["url"], score=0.9)
                )
                add("retrieve", f"Read {summ['title']}: {summ['extract'][:160]}…", 0.85, [summ["url"]])
    else:
        add("retrieve", "Wikipedia unreachable or empty — continuing with local sources.")

    if not sources:
        add("critique", "No sources. Honest gap, not a hallucinated briefing.")
        md = (
            f"# {topic}\n\n"
            "I could not assemble a sourced report. Teach me with `learn this: Title — body`, "
            "drop a file into the vault, or retry when the network can reach Wikipedia.\n"
        )
        chain.answer = md
        chain.confidence = 0.2
        doc = get_desk().write_doc(f"Research: {topic}", md, kind="research")
        return Report(topic, f"Research: {topic}", md, [], steps, 0.2, doc["id"], chain.as_dict(), [])

    add("hypothesize", f"{len(sources)} sources on the table. Prefer agreement; flag lone claims.")
    md, confidence, actions = _synthesize(topic, sources)
    add("critique", "Gaps called out. Next actions are tasks, not slogans.")
    add("decide", f"Commit report · confidence {confidence:.0%} · {len(sources)} sources.")

    chain.answer = md
    chain.confidence = confidence
    title = f"Research: {topic}"
    doc = get_desk().write_doc(title, md, kind="research")
    agi.teach(title, md[:2400])
    tasks = []
    desk = get_desk()
    for act in actions[:5]:
        tasks.append(desk.add_task(act, source="research"))
    return Report(topic, title, md, sources, steps, confidence, doc["id"], chain.as_dict(), tasks)


def _synthesize(topic: str, sources: list[Source]) -> tuple[str, float, list[str]]:
    wiki = [s for s in sources if s.kind == "wikipedia"]
    local = [s for s in sources if s.kind == "knowledge"]
    vault = [s for s in sources if s.kind == "vault"]
    mem = [s for s in sources if s.kind == "memory"]

    verdict_src = wiki[0] if wiki else (local[0] if local else sources[0])
    verdict = _first_sentences(verdict_src.excerpt, 2)

    known_bits = []
    for s in (local + wiki)[:4]:
        known_bits.append(f"- **{s.title}** — {_first_sentences(s.excerpt, 2)}")
    if not known_bits:
        known_bits.append(f"- {_first_sentences(sources[0].excerpt, 2)}")

    evidence = []
    for s in sources[:8]:
        cite = f" ([source]({s.url}))" if s.url else ""
        evidence.append(f"- _{s.kind}_ **{s.title}**{cite}: {_first_sentences(s.excerpt, 1)}")

    gaps = []
    if not wiki:
        gaps.append("No live Wikipedia read — the report is local-only.")
    if not vault:
        gaps.append("Nothing in the vault on this topic. Drop a paper or note to deepen it.")
    if len(sources) < 3:
        gaps.append("Thin source set. Another pass with a tighter query would help.")
    if not gaps:
        gaps.append("Coverage is decent; primary papers and numbers would still sharpen it.")

    actions = [
        f"Read a primary source on {topic}",
        f"Teach CORTEX a tighter article about {topic}",
    ]
    if vault:
        actions.append(f"Cross-check vault file {vault[0].title} against the verdict")
    else:
        actions.append(f"Add a vault file about {topic}")

    conf = min(0.92, 0.42 + 0.1 * len(sources) + (0.12 if wiki else 0) + (0.08 if local else 0))

    md = "\n".join(
        [
            f"# {topic}",
            "",
            f"**Confidence** {conf:.0%} · **{len(sources)} sources**"
            + (f" · Wikipedia: {wiki[0].title}" if wiki else " · local sources only"),
            "",
            "## Verdict",
            verdict,
            "",
            "## What we know",
            "\n".join(known_bits),
            "",
            "## Evidence",
            "\n".join(evidence),
            "",
            "## Gaps",
            "\n".join(f"- {g}" for g in gaps),
            "",
            "## Next actions",
            "\n".join(f"- {a}" for a in actions),
            "",
            "_This report lives in Documents and was taught back into long-term knowledge._",
            "",
        ]
    )
    return md, round(conf, 3), actions


def _first_sentences(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return "(empty)"
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n])[:600]


def _pack_side(agi: "AGI", name: str) -> list[Source]:
    out: list[Source] = []
    for a in agi.knowledge.search(name, k=2):
        out.append(Source("knowledge", a.title, a.body[:900], score=0.85))
    for v in get_workspace().search_vault(name, k=2):
        out.append(Source("vault", v["name"], v.get("excerpt") or "", score=0.75))
    hits = wiki_search(name, k=2)
    if hits:
        summ = wiki_summary(hits[0]["title"])
        if summ:
            out.append(Source("wikipedia", summ["title"], summ["extract"], url=summ["url"], score=0.9))
    return out


def compare_brief(agi: "AGI", user: str) -> Report:
    pair = sides_of(user)
    if not pair:
        return investigate(agi, user)
    left, right = pair
    steps: list[tuple[str, str]] = []
    chain = Chain(question=user, strategy="compare", system=2, confidence=0.6)

    def add(kind: str, text: str, conf: float = 0.75) -> None:
        chain.add(kind, text, conf)
        steps.append((kind, text))

    add("parse", f"Compare “{left}” vs “{right}”. Need primitives, overlap, split, failure modes.")
    add("strategy", "Same sources on both sides. Odysseus compares APIs; I compare ideas.")
    a_src = _pack_side(agi, left)
    b_src = _pack_side(agi, right)
    add("retrieve", f"{left}: {len(a_src)} sources. {right}: {len(b_src)} sources.")

    def blurb(srcs: list[Source], name: str) -> str:
        if not srcs:
            return f"No stored account of {name}."
        return _first_sentences(srcs[0].excerpt, 2)

    md = "\n".join(
        [
            f"# {left} vs {right}",
            "",
            "Odysseus would send this prompt to five vendors. One mind, two characterizations, then the joint and the split.",
            "",
            f"## {left}",
            blurb(a_src, left),
            "",
            f"## {right}",
            blurb(b_src, right),
            "",
            "## Joint",
            "Both are attempts to name a structure. Shared words are not shared mechanisms — check the primitives.",
            "",
            "## Split",
            f"- Domain of {left} vs domain of {right}.",
            "- What each cannot explain without borrowing the other's terms.",
            "- Failure mode: collapsing the pair into a slogan.",
            "",
            "## Evidence",
            "\n".join(
                f"- _{s.kind}_ **{s.title}**" + (f" ([source]({s.url}))" if s.url else "") + f": {_first_sentences(s.excerpt, 1)}"
                for s in (a_src + b_src)[:8]
            )
            or "- none yet",
            "",
            "_Saved as a compare brief and taught back._",
            "",
        ]
    )
    conf = min(0.88, 0.4 + 0.08 * (len(a_src) + len(b_src)))
    add("decide", f"Commit compare brief · conf {conf:.0%}.")
    chain.answer = md
    chain.confidence = conf
    title = f"Compare: {left} vs {right}"
    doc = get_desk().write_doc(title, md, kind="compare")
    agi.teach(title, md[:2000])
    return Report(f"{left} vs {right}", title, md, a_src + b_src, steps, round(conf, 3), doc["id"], chain.as_dict(), [])
