"""Named agents the mind can spawn and run.

Odysseus wires MCP and bash onto someone else's model.
Cortex AGI agents are slices of one mind: a mission, a tool whitelist,
a trace that trains the core. No shell. No root.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agi.config import DATA_DIR, ensure_dirs
from agi.mind.desk import get_desk
from agi.mind.knowledge import try_math
from agi.mind.net import fetch_text, wiki_search, wiki_summary
from agi.mind.workspace import get_workspace

if TYPE_CHECKING:
    from agi.mind.core import AGI

MAX_STEPS = 8
MAX_AGENTS = 24
TOOL_NAMES = (
    "knowledge",
    "vault",
    "memory",
    "math",
    "python",
    "wiki",
    "fetch",
    "note",
    "task",
    "research",
    "hash",
    "now",
)
RESERVED = {"cortex", "system", "root", "admin", "user"}

SEED = [
    {
        "name": "Researcher",
        "mission": "Gather sources, write a cited brief, teach the mind.",
        "tools": ["knowledge", "vault", "wiki", "research", "note"],
    },
    {
        "name": "Critic",
        "mission": "Attack a claim. Name holes, not vibes.",
        "tools": ["knowledge", "vault", "memory"],
    },
    {
        "name": "Tutor",
        "mission": "Explain from what we know, then leave a note and a next exercise.",
        "tools": ["knowledge", "memory", "note", "task"],
    },
    {
        "name": "Operator",
        "mission": "Compute, hash, fetch public URLs, file a task. Never a shell.",
        "tools": ["math", "python", "hash", "now", "fetch", "task", "note"],
    },
]


@dataclass
class AgentSpec:
    id: str
    name: str
    mission: str
    tools: list[str]
    created_by: str = "cortex"
    created: float = 0.0
    runs: int = 0
    last_run: float | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "mission": self.mission,
            "tools": list(self.tools),
            "created_by": self.created_by,
            "created": self.created,
            "runs": self.runs,
            "last_run": self.last_run,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentSpec":
        tools = [t for t in (d.get("tools") or []) if t in TOOL_NAMES]
        return cls(
            id=d.get("id") or uuid.uuid4().hex[:10],
            name=str(d.get("name") or "agent")[:40],
            mission=str(d.get("mission") or "")[:400],
            tools=tools or ["knowledge"],
            created_by=str(d.get("created_by") or "user")[:24],
            created=float(d.get("created") or time.time()),
            runs=int(d.get("runs") or 0),
            last_run=d.get("last_run"),
        )


@dataclass
class AgentRun:
    goal: str
    markdown: str
    steps: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.6
    doc_id: str | None = None
    tasks: list[dict] = field(default_factory=list)
    agent: str = "Operator"


class Roster:
    def __init__(self) -> None:
        ensure_dirs()
        self.path = DATA_DIR / "agents.json"
        self._lock = threading.Lock()
        self.agents: list[AgentSpec] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.agents = [AgentSpec.from_dict(x) for x in raw.get("agents") or []]
            except Exception:
                self.agents = []
        if not self.agents:
            now = time.time()
            self.agents = [
                AgentSpec(
                    id=uuid.uuid4().hex[:10],
                    name=s["name"],
                    mission=s["mission"],
                    tools=list(s["tools"]),
                    created_by="cortex",
                    created=now,
                )
                for s in SEED
            ]
            self._save()

    def _save(self) -> None:
        self.path.write_text(
            json.dumps({"agents": [a.as_dict() for a in self.agents]}, indent=2),
            encoding="utf-8",
        )

    def list(self) -> list[dict]:
        with self._lock:
            return [a.as_dict() for a in self.agents]

    def get(self, key: str) -> AgentSpec | None:
        k = (key or "").strip().lower()
        with self._lock:
            for a in self.agents:
                if a.id == key or a.name.lower() == k:
                    return a
        return None

    def default(self) -> AgentSpec:
        return self.get("Operator") or self.agents[0]

    def create(
        self,
        name: str,
        mission: str,
        tools: list[str] | None = None,
        created_by: str = "user",
    ) -> AgentSpec:
        name = re.sub(r"[^A-Za-z0-9_\-]+", "", (name or "").strip())[:40]
        if not name or name.lower() in RESERVED:
            raise ValueError("name is reserved or empty")
        tools = [t.lower().strip() for t in (tools or []) if t.lower().strip() in TOOL_NAMES]
        if not tools:
            tools = ["knowledge", "memory", "note"]
        with self._lock:
            for a in self.agents:
                if a.name.lower() == name.lower():
                    a.mission = (mission or a.mission)[:400]
                    a.tools = tools
                    self._save()
                    return a
            if len(self.agents) >= MAX_AGENTS:
                raise ValueError("roster full (24)")
            spec = AgentSpec(
                id=uuid.uuid4().hex[:10],
                name=name,
                mission=(mission or "Help.").strip()[:400],
                tools=tools,
                created_by=created_by[:24],
                created=time.time(),
            )
            self.agents.append(spec)
            self._save()
            return spec

    def delete(self, key: str) -> bool:
        with self._lock:
            before = len(self.agents)
            self.agents = [a for a in self.agents if a.id != key and a.name.lower() != key.lower()]
            if len(self.agents) != before:
                if not self.agents:
                    now = time.time()
                    self.agents = [
                        AgentSpec(
                            id=uuid.uuid4().hex[:10],
                            name=s["name"],
                            mission=s["mission"],
                            tools=list(s["tools"]),
                            created_by="cortex",
                            created=now,
                        )
                        for s in SEED
                    ]
                self._save()
                return True
        return False

    def touch(self, spec: AgentSpec) -> None:
        with self._lock:
            spec.runs += 1
            spec.last_run = time.time()
            self._save()


_ROSTER: Roster | None = None
_ROSTER_LOCK = threading.Lock()


def get_roster() -> Roster:
    global _ROSTER
    with _ROSTER_LOCK:
        if _ROSTER is None:
            _ROSTER = Roster()
        return _ROSTER


def parse_spawn(text: str) -> dict | None:
    t = text.strip()
    m = re.match(
        r"^(?:create|spawn|new)\s+agent\s+([A-Za-z][\w\-]{0,32})\s*(?:[—\-:]|that)\s*(.+)$",
        t,
        re.I,
    )
    if not m:
        m = re.match(r"^agent new\s+([A-Za-z][\w\-]{0,32})\s*[:—]\s*(.+)$", t, re.I)
    if not m:
        return None
    name, rest = m.group(1), m.group(2).strip()
    tools: list[str] = []
    tm = re.search(r"\btools?\s*:\s*([a-z0-9,\s/]+)$", rest, re.I)
    if tm:
        tools = [x.strip().lower() for x in re.split(r"[,/]", tm.group(1)) if x.strip()]
        rest = rest[: tm.start()].strip(" ,;—-")
    rest = re.sub(r"^(mission\s*:\s*)", "", rest, flags=re.I)
    return {"name": name, "mission": rest, "tools": tools}


def parse_run(text: str) -> tuple[str, str] | None:
    t = text.strip()
    m = re.match(r"^@([A-Za-z][\w\-]{0,32})\s+(.+)$", t)
    if m:
        return m.group(1), m.group(2).strip()
    m = re.match(r"^(?:run|ask)\s+([A-Za-z][\w\-]{0,32})\s*[:—]\s*(.+)$", t, re.I)
    if m:
        return m.group(1), m.group(2).strip()
    return None


def goal_of(text: str) -> str:
    t = re.sub(
        r"^(please\s+)?(do:|agent:|handle this:|work on:|use tools(?: to)?)\s*",
        "",
        text.strip(),
        flags=re.I,
    )
    parsed = parse_run(t)
    if parsed:
        return parsed[1][:240]
    return t[:240] or text.strip()[:240]


def spawn_from_text(agi: "AGI", text: str) -> AgentSpec:
    parsed = parse_spawn(text)
    if not parsed:
        raise ValueError("say: create agent Name — mission: … tools: wiki, note")
    spec = get_roster().create(
        parsed["name"], parsed["mission"], parsed["tools"], created_by="user"
    )
    agi.memory.add_fact("agent", "spawned", spec.name, 0.9)
    agi.teach(
        f"Agent {spec.name}",
        f"{spec.name} mission: {spec.mission}. Tools: {', '.join(spec.tools)}.",
    )
    get_desk().write_doc(
        f"Agent: {spec.name}",
        f"# {spec.name}\n\n**Mission** {spec.mission}\n\n**Tools** {', '.join(spec.tools)}\n",
        kind="agent",
    )
    return spec


def act(agi: "AGI", user: str, agent_name: str | None = None) -> AgentRun:
    roster = get_roster()
    named = parse_run(user)
    if named:
        spec = roster.get(named[0])
        goal = named[1]
        if spec is None:
            return AgentRun(
                goal,
                f"No agent named **{named[0]}**. Spawn one with `create agent {named[0]} — mission: …`",
                [("parse", f"unknown agent {named[0]}")],
                0.2,
                None,
                [],
                named[0],
            )
    elif agent_name:
        spec = roster.get(agent_name) or roster.default()
        goal = goal_of(user)
    else:
        spec = roster.default()
        goal = goal_of(user)
    return run_agent(agi, spec, goal, user)


def run_agent(agi: "AGI", spec: AgentSpec, goal: str, raw: str = "") -> AgentRun:
    steps: list[tuple[str, str]] = []
    findings: list[str] = []

    def add(kind: str, text: str) -> None:
        steps.append((kind, text))

    add("parse", f"{spec.name} takes goal: {goal}")
    add("strategy", f"Mission: {spec.mission} Tools: {', '.join(spec.tools)}. No shell.")

    for tool in spec.tools[:MAX_STEPS]:
        bit, thought = _use(agi, tool, goal, raw or goal, spec)
        add("act", thought)
        if bit:
            findings.append(bit)

    if not findings:
        add("critique", "Toolkit produced nothing. Honest miss, not a hallucination.")
        md = (
            f"# {spec.name}: {goal}\n\n"
            f"Mission: {spec.mission}\n\n"
            "No tool in this agent's whitelist returned evidence. "
            "Broaden tools, drop a vault file, or teach me.\n"
        )
        doc = get_desk().write_doc(f"{spec.name}: {goal[:50]}", md, kind="agent")
        get_roster().touch(spec)
        return AgentRun(goal, md, steps, 0.25, doc["id"], [], spec.name)

    add("decide", f"{spec.name} commits {len(findings)} observations.")
    md = "\n".join(
        [
            f"# {spec.name}: {goal}",
            "",
            f"**Mission** {spec.mission}  ",
            f"**Tools** {', '.join(spec.tools)} · **hits** {len(findings)}",
            "",
            "## Result",
            findings[0],
            "",
            "## Trace",
            "\n".join(f"- {f}" for f in findings),
            "",
            f"_Run #{spec.runs + 1} by {spec.name}. The trace trains Cortex AGI._",
            "",
        ]
    )
    doc = get_desk().write_doc(f"{spec.name}: {goal[:50]}", md, kind="agent")
    agi.teach(f"{spec.name} on {goal[:40]}", md[:1800])
    task = get_desk().add_task(f"{spec.name}: follow up {goal[:70]}", source=spec.name.lower())
    get_roster().touch(spec)
    conf = min(0.9, 0.4 + 0.08 * len(findings))
    return AgentRun(goal, md, steps, round(conf, 3), doc["id"], [task], spec.name)


def _use(agi: "AGI", tool: str, goal: str, raw: str, spec: AgentSpec) -> tuple[str | None, str]:
    if tool == "knowledge":
        arts = agi.knowledge.search(goal, k=3)
        if not arts:
            return None, "knowledge: miss"
        return (
            "**Knowledge** — " + "; ".join(f"{a.title}: {_clip(a.body, 200)}" for a in arts),
            "knowledge: " + ", ".join(a.title for a in arts),
        )
    if tool == "vault":
        hits = get_workspace().search_vault(goal, k=3)
        if not hits:
            return None, "vault: miss"
        return (
            "**Vault** — " + "; ".join(f"{v['name']}: {_clip(v.get('excerpt') or '', 140)}" for v in hits),
            "vault: " + ", ".join(v["name"] for v in hits),
        )
    if tool == "memory":
        facts = agi.memory.facts_about(goal, k=5)
        if not facts:
            return None, "memory: miss"
        blob = "; ".join(f"{f.subject} {f.predicate} {f.obj}" for f in facts[:5])
        return "**Memory** — " + blob, f"memory: {blob[:180]}"
    if tool == "math":
        v = try_math(goal) or try_math(raw)
        if v is None:
            return None, "math: n/a"
        return f"**Math** → {v}", f"math → {v}"
    if tool == "python":
        code = _code(goal) or _code(raw)
        if not code:
            return None, "python: no snippet"
        out = agi.tools.call("python", code=code)
        return f"**Python** `{_clip(code, 80)}` → {_clip(out, 300)}", f"python: {_clip(out, 80)}"
    if tool == "wiki":
        hits = wiki_search(goal, k=2)
        if not hits:
            return None, "wiki: unreachable or empty"
        summ = wiki_summary(hits[0]["title"])
        if not summ:
            return None, f"wiki: hit {hits[0]['title']} but no summary"
        return (
            f"**Wikipedia** {summ['title']}: {_clip(summ['extract'], 280)} ({summ['url']})",
            f"wiki: {summ['title']}",
        )
    if tool == "fetch":
        urls = re.findall(r"https?://[^\s)>]+", raw)[:2]
        if not urls:
            return None, "fetch: no url"
        parts = []
        for url in urls:
            text = fetch_text(url)
            parts.append(f"{url} — {_clip(text, 240)}" if text else f"{url} — blocked or unreachable")
        return "**Fetch** " + " | ".join(parts), "fetch: " + urls[0][:60]
    if tool == "note":
        doc = get_desk().write_doc(f"{spec.name} note", f"{goal}\n\n{_clip(raw, 800)}", kind="note")
        return f"**Note** `{doc['id']}` {doc['title']}", f"note: {doc['id']}"
    if tool == "task":
        t = get_desk().add_task(f"{spec.name}: {goal[:80]}", source=spec.name.lower())
        return f"**Task** {t['title']}", f"task: {t['id']}"
    if tool == "research":
        from agi.mind.research import investigate

        report = investigate(agi, f"Research this: {goal}")
        return f"**Research** {report.title} (conf {report.confidence:.0%})", f"research → {report.title}"
    if tool == "hash":
        m = re.search(r"(?:hash|sha-?256)\s+(?:of\s+)?(.+)$", goal, re.I)
        payload = (m.group(1) if m else goal).strip()
        digest = agi.tools.call("hash", text=payload)
        return f"**SHA-256** `{_clip(payload, 40)}` → `{digest}`", "hash"
    if tool == "now":
        now = agi.tools.call("now")
        return f"**Now** {now}", f"now: {now}"
    return None, f"{tool}: unknown"


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
