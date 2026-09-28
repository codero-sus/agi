"""A bounded tool loop. Odysseus shells out; CORTEX acts, then keeps the trace."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agi.mind.desk import get_desk
from agi.mind.knowledge import try_math
from agi.mind.net import fetch_text
from agi.mind.workspace import get_workspace

if TYPE_CHECKING:
    from agi.mind.core import AGI

MAX_STEPS = 6


@dataclass
class AgentRun:
    goal: str
    markdown: str
    steps: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.6
    doc_id: str | None = None
    tasks: list[dict] = field(default_factory=list)


def goal_of(text: str) -> str:
    t = re.sub(r"^(please\s+)?(do:|agent:|handle this:|work on:|use tools(?: to)?)\s*", "", text.strip(), flags=re.I)
    return t[:240] or text.strip()[:240]


def act(agi: "AGI", user: str) -> AgentRun:
    goal = goal_of(user)
    steps: list[tuple[str, str]] = []
    findings: list[str] = []

    def add(kind: str, text: str) -> None:
        steps.append((kind, text))

    add("parse", f"Goal: {goal}. Plan → tools → observe → keep. No shell.")
    add("strategy", "Knowledge, vault, memory, math, public URLs. Write a trace the mind can reuse.")

    math_v = try_math(goal)
    if math_v is None and re.search(r"\b(compute|calculate|eval|python)\b", goal, re.I):
        code = _code(goal)
        if code:
            add("act", f"python: {code[:80]}")
            out = agi.tools.call("python", code=code)
            findings.append(f"**Python** `{code}` → {out[:400]}")
            math_v = out
    elif math_v is not None:
        add("act", f"math → {math_v}")
        findings.append(f"**Math** → {math_v}")

    arts = agi.knowledge.search(goal, k=3)
    if arts:
        add("act", "knowledge: " + ", ".join(a.title for a in arts))
        findings.append("**Knowledge** — " + "; ".join(f"{a.title}: {_clip(a.body, 220)}" for a in arts))
    else:
        add("act", "knowledge: miss")

    vault = get_workspace().search_vault(goal, k=3)
    if vault:
        add("act", "vault: " + ", ".join(v["name"] for v in vault))
        findings.append("**Vault** — " + "; ".join(f"{v['name']}: {_clip(v.get('excerpt') or '', 160)}" for v in vault))

    facts = agi.memory.facts_about(goal, k=5)
    if facts:
        blob = "; ".join(f"{f.subject} {f.predicate} {f.obj}" for f in facts[:5])
        add("act", f"memory: {blob[:200]}")
        findings.append("**Memory** — " + blob)

    for url in re.findall(r"https?://[^\s)>]+", user)[:2]:
        add("act", f"fetch {url[:80]}")
        text = fetch_text(url)
        if text:
            findings.append(f"**Fetch** {url} — {_clip(text, 280)}")
        else:
            findings.append(f"**Fetch** {url} — blocked or unreachable (private hosts are refused).")

    if not findings:
        add("critique", "No tool produced evidence. I will not invent a result.")
        md = (
            f"# Agent: {goal}\n\n"
            "No knowledge, vault file, memory, or computation matched. "
            "Drop a file, `learn this:`, or be more specific.\n"
        )
        doc = get_desk().write_doc(f"Agent: {goal[:60]}", md, kind="agent")
        return AgentRun(goal, md, steps, 0.25, doc["id"], [])

    add("decide", f"{len(findings)} observations. Write the trace; spawn a next action.")
    md = "\n".join(
        [
            f"# Agent: {goal}",
            "",
            f"**Observations** {len(findings)} · no shell · no MCP",
            "",
            "## Result",
            findings[0],
            "",
            "## Trace",
            "\n".join(f"- {f}" for f in findings),
            "",
            "_This run is a document. Odysseus would have reached for bash; I kept the work._",
            "",
        ]
    )
    title = f"Agent: {goal[:60]}"
    doc = get_desk().write_doc(title, md, kind="agent")
    agi.teach(title, md[:1800])
    task = get_desk().add_task(f"Review agent run: {goal[:80]}", source="agent")
    conf = min(0.9, 0.45 + 0.1 * len(findings))
    return AgentRun(goal, md, steps, round(conf, 3), doc["id"], [task])


def _clip(text: str, n: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t if len(t) <= n else t[: n - 1] + "…"


def _code(goal: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", goal, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:python|compute|eval|calculate)\s*:?\s*(.+)$", goal, re.I)
    if m:
        return m.group(1).strip()
    return ""
