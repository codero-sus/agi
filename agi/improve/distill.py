"""Train CortexGPT from the mind, and bake that mind into Ollama / LLMHoster.

Ollama gets a derived model (`cortex-agi`) via /api/create (Modelfile FROM+SYSTEM+MESSAGE).
LLMHoster gets a JSONL corpus plus a fine-tune POST if the hoster exposes one.
Both can teach CortexGPT by distillation. Neither replaces local Adam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agi.config import DATA_DIR, ensure_dirs
from agi.inference.remote import RemoteChat, _json, list_models
from agi.mind.hosts import get_hosts

if TYPE_CHECKING:
    from agi.mind.core import AGI

BAKE_NAME = "cortex-agi"


def mind_corpus(agi: "AGI", turns: int = 48) -> str:
    parts = [agi.identity.system_preamble()]
    facts = agi.memory.all_facts(k=24)
    if facts:
        parts.append("Facts:\n" + "\n".join(f"- {f.subject} {f.predicate} {f.obj}" for f in facts))
    arts = agi.knowledge.taught[-8:] or agi.knowledge.search("agi", k=4)
    for a in arts:
        parts.append(f"{a.title}. {a.body[:600]}")
    dlg = agi.memory.recent_dialogue(k=turns)
    if dlg:
        parts.append("\n".join(f"{d['role']}: {d['content'][:400]}" for d in dlg))
    return "\n\n".join(parts)[:12000]


def _pairs(agi: "AGI") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    dlg = agi.memory.recent_dialogue(k=40)
    user = ""
    for d in dlg:
        if d.get("role") == "user":
            user = (d.get("content") or "")[:500]
        elif d.get("role") == "agi" and user:
            out.append((user, (d.get("content") or "")[:800]))
            user = ""
    for f in agi.memory.all_facts(k=8):
        q = f"What do you know: {f.subject} {f.predicate}?"
        a = f"{f.subject} {f.predicate} {f.obj}."
        out.append((q, a))
    return out[-24:]


def _prompts(agi: "AGI") -> list[str]:
    qs = []
    arts = agi.knowledge.taught[-6:] or agi.knowledge.search("mind", k=4)
    for a in arts:
        qs.append(f"In one short paragraph, teach: {a.title}")
    for f in agi.memory.all_facts(k=4):
        qs.append(f"State this as a sentence: {f.subject} {f.predicate} {f.obj}")
    if not qs:
        qs = ["Who are you, in two sentences?", "How do you improve yourself?"]
    return qs[:6]


def _write_jsonl(pairs: list[tuple[str, str]]) -> str:
    ensure_dirs()
    path = DATA_DIR / "train.jsonl"
    import json as _j

    lines = [
        _j.dumps(
            {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]},
            ensure_ascii=False,
        )
        for u, a in pairs
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (DATA_DIR / "train.txt").write_text(
        "\n\n".join(f"<SFT>\nuser: {u}\nagi: {a}" for u, a in pairs), encoding="utf-8"
    )
    return str(path)


def _first_model(kind: str, url: str, key: str, current: str) -> str:
    if current:
        return current
    try:
        models = list_models(kind, url, key)
    except Exception:
        return ""
    return models[0] if models else ""


def _bake_ollama(agi: "AGI") -> dict:
    h = get_hosts()
    cfg = h.data["ollama"]
    url = (cfg.get("url") or "http://127.0.0.1:11434").rstrip("/")
    base_model = _first_model("ollama", url, "", cfg.get("model") or "")
    if not base_model:
        return {"ok": False, "error": "pick an Ollama model first (probe, then choose)"}
    chat = []
    for u, a in _pairs(agi)[:8]:
        chat.append({"role": "user", "content": u})
        chat.append({"role": "assistant", "content": a})
    payload: dict = {
        "model": BAKE_NAME,
        "from": base_model,
        "system": agi.identity.system_preamble()[:4000],
        "stream": False,
    }
    if chat:
        payload["messages"] = chat
    try:
        code, data = _json(url + "/api/create", "ollama", "", payload, timeout=180)
    except Exception as e:
        return {"ok": False, "error": str(e), "from": base_model}
    err = None
    if isinstance(data, dict):
        err = data.get("error")
        status = data.get("status")
    else:
        status = None
    if code in (200, 201) and not err:
        with h._lock:
            h.data["ollama"]["model"] = BAKE_NAME
            h._save()
        h.bind(agi.engine)
        return {"ok": True, "model": BAKE_NAME, "from": base_model, "status": status or "created"}
    return {"ok": False, "error": f"ollama create {code}: {err or data}", "from": base_model}


def _bake_hoster(agi: "AGI", pairs: list[tuple[str, str]]) -> dict:
    h = get_hosts()
    cfg = h.data["hoster"]
    url = (cfg.get("url") or "http://127.0.0.1:8624").rstrip("/")
    key = cfg.get("key") or ""
    model = cfg.get("model") or ""
    text = "\n\n".join(f"user: {u}\nagi: {a}" for u, a in pairs)[:80000]
    body = {"model": model or "default", "training_data": text, "hyperparameters": {"n_epochs": 1}}
    last = "no train route"
    for path in ("/v1/fine_tuning/jobs", "/train", "/v1/train", "/api/train"):
        try:
            code, data = _json(url + path, "hoster", key, body, timeout=4)
        except Exception as e:
            last = str(e)
            break
        if code in (200, 201, 202):
            return {"ok": True, "path": path, "model": model, "file": str(DATA_DIR / "train.jsonl")}
        if code in (401, 403):
            return {"ok": False, "error": "hoster auth failed", "file": str(DATA_DIR / "train.jsonl")}
        last = f"{path} → {code}"
        if isinstance(data, dict) and data.get("error"):
            last = f"{path} → {data.get('error')}"
        if code == 0:
            break
    return {
        "ok": False,
        "error": f"hoster has no train API ({last}); wrote data/train.jsonl for llama.cpp finetune",
        "file": str(DATA_DIR / "train.jsonl"),
    }


def _teacher(agi: "AGI", kind: str) -> dict:
    h = get_hosts()
    with h._lock:
        if kind == "ollama":
            cfg = dict(h.data["ollama"])
            remote = RemoteChat("ollama", cfg.get("url") or "http://127.0.0.1:11434", cfg.get("model") or "")
        else:
            cfg = dict(h.data["hoster"])
            remote = RemoteChat(
                "hoster",
                cfg.get("url") or "http://127.0.0.1:8624",
                cfg.get("model") or "",
                cfg.get("key") or "",
            )
    n = 0
    errors: list[str] = []
    for q in _prompts(agi)[:2]:
        try:
            ans = remote.generate(q, max_new=120, temperature=0.4)
        except Exception as e:
            errors.append(str(e)[:160])
            break
        if ans and len(ans) > 12:
            agi.engine.train_on(f"{q}\n{ans}", steps=4, blocking=False)
            n += 1
    return {"ok": n > 0, "samples": n, "error": errors[0] if errors and not n else None}


def _ok_row(row: dict) -> bool:
    if row.get("ok"):
        return True
    dist = row.get("distill") or {}
    return bool(dist.get("ok") or dist.get("samples"))


def run_train(agi: "AGI", targets: list[str] | None = None) -> dict:
    """targets: local, ollama, hoster. Default: all three."""
    want = [t for t in (targets or ["local", "ollama", "hoster"]) if t in {"local", "ollama", "hoster"}]
    if not want:
        want = ["local", "ollama", "hoster"]
    pairs = _pairs(agi)
    corpus = mind_corpus(agi)
    file_path = _write_jsonl(pairs)
    out: dict = {"ok": True, "file": file_path, "pairs": len(pairs), "targets": want, "results": {}}
    if "local" in want:
        agi.engine.train_on(corpus, steps=8, blocking=False)
        out["results"]["local"] = {"ok": True, "queued": 8, "note": "CortexGPT background Adam"}
    if "ollama" in want:
        baked = _bake_ollama(agi)
        taught = _teacher(agi, "ollama")
        out["results"]["ollama"] = {**baked, "distill": taught}
    if "hoster" in want:
        baked = _bake_hoster(agi, pairs)
        taught = _teacher(agi, "hoster")
        out["results"]["hoster"] = {**baked, "distill": taught}
    out["ok"] = any(_ok_row(r) for r in out["results"].values()) if out["results"] else True
    return out
