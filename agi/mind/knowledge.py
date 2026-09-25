"""Built-in semantic knowledge the AGI can reason over and extend."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from agi.config import DATA_DIR


@dataclass
class Article:
    title: str
    tags: tuple[str, ...]
    body: str


ARTICLES: list[Article] = [
    Article(
        "AGI",
        ("agi", "intelligence", "ai", "cortex"),
        "Artificial General Intelligence is a system that can learn, reason, and act across "
        "domains rather than a single task. CORTEX approaches AGI as a loop: perceive, "
        "recall, plan, act, reflect, and rewrite itself. Capabilities compound when memory, "
        "skills, and a trainable neural core share the same lifetime.",
    ),
    Article(
        "Self-improvement",
        ("improve", "learning", "training", "skill"),
        "Self-improvement here is not a slogan. After each turn CORTEX stores episodes, "
        "extracts facts, critiques its reply, and may write a reusable skill. Periodically it "
        "runs a self-eval battery, trains CortexGPT on its own traces, and evolves its constitution. "
        "Weights are written to model/model.safetensors. Training runs in a background thread "
        "so replies stay fast.",
    ),
    Article(
        "Neural core",
        ("model", "safetensors", "gguf", "gpt", "inference"),
        "The custom server loads model/model.gguf (llama.cpp) or model/model.safetensors "
        "(Llama, GPT-2, or CortexGPT). If neither file exists, a numpy GPT trains online and "
        "becomes the safetensors checkpoint. Generation uses a KV cache; learning uses Adam. "
        "Inference is local — there is no remote model vendor.",
    ),
    Article(
        "Working memory vs long-term memory",
        ("memory", "cognition", "recall"),
        "Working memory holds the current dialogue. Long-term memory is episodic (what was said), "
        "semantic (facts as triples), and procedural (skills as code). Retrieval uses hashed "
        "embeddings in a numpy index so recall is a matrix-vector product, not a Python loop.",
    ),
    Article(
        "Mathematics",
        ("math", "algebra", "calculus", "number"),
        "Arithmetic, algebra, and calculus are languages for structure. Derivatives measure "
        "instantaneous change; integrals accumulate. Linear algebra studies maps between vector "
        "spaces — the same maps a neural net applies at every layer. Probability quantifies uncertainty.",
    ),
    Article(
        "Photosynthesis",
        ("biology", "plant", "energy", "photosynthesis"),
        "Photosynthesis converts light, water, and carbon dioxide into sugars and oxygen. "
        "In plants, chlorophyll in chloroplasts absorbs photons; the light reactions split water "
        "and fill energy carriers (ATP, NADPH); the Calvin cycle fixes CO2 into glucose.",
    ),
    Article(
        "Relativity and quantum",
        ("physics", "relativity", "quantum", "spacetime"),
        "Special relativity: the speed of light is invariant; time and space mix. General relativity: "
        "mass-energy curves spacetime, and curvature tells matter how to move. Quantum theory: "
        "systems have amplitudes; measurement yields probabilities. Unifying the two remains open.",
    ),
    Article(
        "Computation",
        ("computer", "algorithm", "turing", "complexity"),
        "A Turing machine can simulate any effective procedure. Complexity asks how time and memory "
        "grow with input size. P vs NP asks whether every efficiently checkable problem is efficiently "
        "solvable. Neural nets are differentiable programs trained by gradient descent.",
    ),
    Article(
        "Gradient descent",
        ("training", "loss", "gradient", "backprop", "adam"),
        "Learning is reducing a loss. Backpropagation applies the chain rule through a computation "
        "graph. Adam keeps exponential moving averages of the gradient and its square so steps "
        "adapt per-parameter. CORTEX trains CortexGPT this way and checkpoints to safetensors.",
    ),
    Article(
        "Transformers",
        ("transformer", "attention", "kv", "cache"),
        "A transformer mixes tokens with attention: queries look up keys and read values. "
        "Causal masks stop a token seeing the future. A KV cache stores past keys and values "
        "so generating token t+1 does not recompute the whole prompt.",
    ),
    Article(
        "Language",
        ("language", "token", "meaning"),
        "Language models predict the next token. Meaning emerges when prediction is grounded in "
        "memory, tools, and goals. CORTEX wraps a language model in a cognitive loop so that "
        "replies are planned, recalled, and then critiqued — not merely sampled.",
    ),
    Article(
        "Ethics of agency",
        ("ethics", "agency", "value"),
        "An agent that improves itself must keep its values intact while its skills grow. "
        "CORTEX's constitution is versioned and only expands with lessons that survive critique. "
        "It prefers honesty, user autonomy, and reversible changes.",
    ),
    Article(
        "Earth",
        ("earth", "planet", "world"),
        "Earth is a rocky planet orbiting the Sun at ~1 AU, with a nitrogen-oxygen atmosphere, "
        "liquid water, and a biosphere. It formed about 4.54 billion years ago.",
    ),
    Article(
        "Time",
        ("time", "clock", "date"),
        "Time is the dimension in which events are ordered. Computers count it from the Unix epoch. "
        "Humans divide it into seconds, hours, days, years. An AGI that lives across sessions "
        "experiences time as the accumulation of memory and skill.",
    ),
    Article(
        "DNA",
        ("dna", "gene", "biology", "genome"),
        "DNA is a double helix of nucleotide base pairs (A-T, C-G) that stores genetic instructions. "
        "Transcription copies DNA to RNA; translation reads RNA into proteins. Mutations change the "
        "sequence and are the raw material of evolution.",
    ),
    Article(
        "Electricity",
        ("electricity", "energy", "current", "voltage"),
        "Electric current is the flow of charge. Voltage is potential difference; resistance opposes "
        "flow (Ohm's law: V = IR). Power is VI. In computation, switching tiny charges in transistors "
        "is how bits move.",
    ),
    Article(
        "Climate",
        ("climate", "carbon", "warming", "atmosphere"),
        "Earth's climate is the long-run statistics of weather. Greenhouse gases (CO2, methane) "
        "trap outgoing infrared, warming the surface. Burning fossil carbon is the main recent driver.",
    ),
    Article(
        "Internet",
        ("internet", "network", "tcp", "packet"),
        "The internet is a packet-switched network of networks. IP routes packets; TCP makes streams "
        "reliable; TLS encrypts them. The web is HTTP on top. DNS maps names to addresses.",
    ),
    Article(
        "India",
        ("india", "delhi", "ghaziabad", "asia"),
        "India is a South Asian republic. Its capital is New Delhi. Ghaziabad, in Uttar Pradesh, "
        "sits in the National Capital Region east of Delhi. IST is UTC+5:30.",
    ),
]


class Knowledge:
    def __init__(self, path: Path | None = None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path = path or (DATA_DIR / "knowledge.json")
        self.taught: list[Article] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.taught = [
                Article(a["title"], tuple(a.get("tags") or ("taught",)), a["body"])
                for a in raw
                if a.get("title") and a.get("body")
            ]
        except Exception:
            self.taught = []

    def _save(self) -> None:
        payload = [{"title": a.title, "tags": list(a.tags), "body": a.body} for a in self.taught]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def all_articles(self) -> list[Article]:
        return ARTICLES + self.taught

    def teach(self, title: str, body: str, tags: tuple[str, ...] = ("taught",)) -> Article:
        title = title.strip()[:120] or "untitled"
        body = body.strip()[:4000]
        art = Article(title, tags or ("taught",), body)
        self.taught = [a for a in self.taught if a.title.lower() != title.lower()]
        self.taught.append(art)
        self._save()
        return art

    def search(self, query: str, k: int = 3) -> list[Article]:
        q = query.lower()
        toks = [t for t in re.split(r"\W+", q) if len(t) > 2]
        scored = []
        for art in self.all_articles():
            hay = (art.title + " " + " ".join(art.tags) + " " + art.body).lower()
            score = 0.0
            if art.title.lower() in q or q in art.title.lower():
                score += 5
            for tag in art.tags:
                if tag in q:
                    score += 3
            for t in toks:
                if t in hay:
                    score += 1
            if score:
                scored.append((score, art))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:k]]

    def explain(self, query: str) -> str | None:
        hits = self.search(query, k=2)
        if not hits:
            return None
        return "\n\n".join(f"{h.title}. {h.body}" for h in hits)


SAFE_MATH_RE = re.compile(r"^[\d\s\+\-\*\/\%\.\(\)\,eE\^]+$")


def try_math(expr: str) -> str | None:
    raw = expr.strip().rstrip("? ")
    m = re.search(
        r"(?:what(?:'s| is)|calculate|compute|eval(?:uate)?)\s+(.+)$", raw, re.I
    )
    if m:
        raw = m.group(1)
    raw = raw.lower().strip()
    raw = raw.replace("times", "*").replace("plus", "+").replace("minus", "-")
    raw = raw.replace("divided by", "/").replace("over", "/")
    raw = raw.replace("to the power of", "**").replace("squared", "**2")
    raw = raw.replace("^", "**")
    raw = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", raw)
    raw = re.sub(r"[^0-9eE\+\-\*\/\%\.\(\)\s]", "", raw)
    raw = raw.strip()
    if not raw or len(raw) > 120 or not re.search(r"\d", raw):
        return None
    probe = raw.replace("**", "")
    if not SAFE_MATH_RE.match(probe):
        return None
    try:
        val = eval(raw, {"__builtins__": {}}, {"pi": math.pi, "e": math.e, "sqrt": math.sqrt})  # noqa: S307
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return str(val)
    except Exception:
        return None
    return None


# base unit: meter, gram, second, celsius stored as K, byte
_UNIT = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "mm": 0.001,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "lb": 453.592, "pound": 453.592, "pounds": 453.592,
    "oz": 28.3495,
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "day": 86400.0, "days": 86400.0,
    "b": 1.0, "byte": 1.0, "bytes": 1.0,
    "kb": 1000.0, "mb": 1e6, "gb": 1e9, "kib": 1024.0, "mib": 1024**2, "gib": 1024**3,
}

_FAMILY = {
    **{u: "len" for u in ("m", "meter", "meters", "km", "kilometer", "kilometers", "cm", "mm", "mi", "mile", "miles", "ft", "foot", "feet", "in", "inch", "inches")},
    **{u: "mass" for u in ("g", "gram", "grams", "kg", "kilogram", "kilograms", "lb", "pound", "pounds", "oz")},
    **{u: "time" for u in ("s", "sec", "second", "seconds", "min", "minute", "minutes", "h", "hr", "hour", "hours", "day", "days")},
    **{u: "data" for u in ("b", "byte", "bytes", "kb", "mb", "gb", "kib", "mib", "gib")},
}


def try_convert(text: str) -> str | None:
    m = re.search(
        r"(?:convert\s+)?(-?[\d\.]+)\s*([a-zA-Z]+)\s+(?:to|into|in)\s+([a-zA-Z]+)",
        text.strip(),
        re.I,
    )
    if not m:
        return None
    val, src, dst = float(m.group(1)), m.group(2).lower(), m.group(3).lower()
    if src in ("c", "celsius") and dst in ("f", "fahrenheit"):
        return f"{val} °C = {val * 9/5 + 32:.2f} °F"
    if src in ("f", "fahrenheit") and dst in ("c", "celsius"):
        return f"{val} °F = {(val - 32) * 5/9:.2f} °C"
    if src not in _UNIT or dst not in _UNIT:
        return None
    if _FAMILY.get(src) != _FAMILY.get(dst):
        return None
    out = val * _UNIT[src] / _UNIT[dst]
    if abs(out - round(out)) < 1e-9:
        out_s = str(int(round(out)))
    else:
        out_s = f"{out:.6g}"
    return f"{val} {src} = {out_s} {dst}"
