"""Built-in semantic knowledge the AGI can reason over and extend."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


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
        "Weights are written to model/model.safetensors.",
    ),
    Article(
        "Neural core",
        ("model", "safetensors", "gguf", "gpt", "inference"),
        "The custom server loads model/model.gguf (llama.cpp) or model/model.safetensors "
        "(Llama, GPT-2, or CortexGPT). If neither file exists, a numpy GPT trains online and "
        "becomes the safetensors checkpoint. Inference is local — there is no remote model vendor.",
    ),
    Article(
        "Working memory vs long-term memory",
        ("memory", "cognition", "recall"),
        "Working memory holds the current dialogue. Long-term memory is episodic (what was said), "
        "semantic (facts as triples), and procedural (skills as code). Retrieval uses hashed "
        "embeddings and keyword overlap.",
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
        ("training", "loss", "gradient", "backprop"),
        "Learning is reducing a loss. Backpropagation applies the chain rule through a computation "
        "graph. Stochastic gradient descent updates weights from mini-batches. CORTEX's own GPT "
        "does this in numpy and checkpoints to safetensors.",
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
]


class Knowledge:
    def search(self, query: str, k: int = 3) -> list[Article]:
        q = query.lower()
        toks = [t for t in re.split(r"\W+", q) if len(t) > 2]
        scored = []
        for art in ARTICLES:
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
    raw = raw.replace("^", "**").replace("x", "*")
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
