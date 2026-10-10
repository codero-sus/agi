"""Which mouth Cortex AGI speaks through: local core, Ollama, OpenRouter, LLMHoster."""

from __future__ import annotations

import json
import os
import threading
from typing import TYPE_CHECKING

from agi.config import DATA_DIR, ensure_dirs
from agi.inference.remote import RemoteChat, _host_ok, probe

if TYPE_CHECKING:
    from agi.inference.loader import ModelEngine

KINDS = ("local", "ollama", "hoster", "openrouter")

DEFAULTS = {
    "active": "local",
    "ollama": {"url": os.environ.get("AGI_OLLAMA_URL", "http://127.0.0.1:11434"), "model": os.environ.get("AGI_OLLAMA_MODEL", "")},
    "hoster": {
        "url": os.environ.get("AGI_HOSTER_URL", "http://127.0.0.1:8624"),
        "model": os.environ.get("AGI_HOSTER_MODEL", ""),
        "key": os.environ.get("AGI_HOSTER_KEY") or os.environ.get("CORTEX_API_KEY") or "",
    },
    "openrouter": {
        "model": os.environ.get("AGI_OPENROUTER_MODEL", "openrouter/auto"),
        "key": os.environ.get("AGI_OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY") or "",
    },
}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _mask(k: str) -> str:
    k = k or ""
    if not k:
        return ""
    if len(k) <= 8:
        return "••••"
    return k[:4] + "…" + k[-2:]


class Hosts:
    def __init__(self) -> None:
        ensure_dirs()
        self.path = DATA_DIR / "hosts.json"
        self._lock = threading.Lock()
        self.data = json.loads(json.dumps(DEFAULTS))
        self._load()
        env_pick = (os.environ.get("AGI_LLM") or "").strip().lower()
        if env_pick in KINDS and self.data.get("active") == "local":
            self.data["active"] = env_pick

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    if raw.get("active") in KINDS:
                        self.data["active"] = raw["active"]
                    for k in ("ollama", "hoster", "openrouter"):
                        if isinstance(raw.get(k), dict):
                            self.data[k].update({kk: vv for kk, vv in raw[k].items() if vv is not None})
            except Exception:
                pass
        # env keys win if file has none
        if not self.data["openrouter"].get("key"):
            self.data["openrouter"]["key"] = DEFAULTS["openrouter"]["key"]
        if not self.data["hoster"].get("key"):
            self.data["hoster"]["key"] = DEFAULTS["hoster"]["key"]

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def snapshot(self) -> dict:
        with self._lock:
            d = json.loads(json.dumps(self.data))
        d["openrouter"]["key"] = _mask(d["openrouter"].get("key") or "")
        d["openrouter"]["has_key"] = bool(self.data["openrouter"].get("key"))
        d["hoster"]["key"] = _mask(d["hoster"].get("key") or "")
        d["hoster"]["has_key"] = bool(self.data["hoster"].get("key"))
        d["kinds"] = list(KINDS)
        return d

    def patch(self, body: dict) -> dict:
        with self._lock:
            active = body.get("active")
            if active in KINDS:
                self.data["active"] = active
            for k in ("ollama", "hoster", "openrouter"):
                if not isinstance(body.get(k), dict):
                    continue
                src = body[k]
                dst = self.data[k]
                if "url" in src and src["url"] and k != "openrouter":
                    url = str(src["url"]).strip()[:200]
                    if _host_ok(url, k):
                        dst["url"] = url
                if "model" in src:
                    dst["model"] = str(src["model"] or "").strip()[:120]
                if "key" in src:
                    key = str(src["key"] or "").strip()
                    if key and "…" not in key and not key.startswith("•"):
                        dst["key"] = key[:200]
            self._save()
        return self.snapshot()

    def client(self) -> RemoteChat | None:
        with self._lock:
            active = self.data.get("active") or "local"
            if active == "local":
                return None
            if active == "ollama":
                cfg = self.data["ollama"]
                return RemoteChat("ollama", cfg.get("url") or DEFAULTS["ollama"]["url"], cfg.get("model") or "")
            if active == "hoster":
                cfg = self.data["hoster"]
                return RemoteChat(
                    "hoster",
                    cfg.get("url") or DEFAULTS["hoster"]["url"],
                    cfg.get("model") or "",
                    cfg.get("key") or "",
                )
            if active == "openrouter":
                cfg = self.data["openrouter"]
                return RemoteChat(
                    "openrouter",
                    OPENROUTER_BASE,
                    cfg.get("model") or "openrouter/auto",
                    cfg.get("key") or "",
                )
        return None

    def bind(self, engine: "ModelEngine") -> None:
        engine.bind_remote(self.client())

    def probe_all(self) -> dict:
        with self._lock:
            ollama_url = self.data["ollama"]["url"]
            hoster_url = self.data["hoster"]["url"]
            hoster_key = self.data["hoster"].get("key") or ""
            or_key = self.data["openrouter"].get("key") or ""
        out = {
            "local": {"ok": True, "kind": "local", "models": ["cortex-gpt"], "error": None},
            "ollama": probe("ollama", ollama_url),
            "hoster": probe("hoster", hoster_url, hoster_key),
            "openrouter": probe("openrouter", OPENROUTER_BASE, or_key) if or_key else {"ok": False, "kind": "openrouter", "models": [], "error": "no API key", "ms": 0},
        }
        return out


_HOSTS: Hosts | None = None
_LOCK = threading.Lock()


def get_hosts() -> Hosts:
    global _HOSTS
    with _LOCK:
        if _HOSTS is None:
            _HOSTS = Hosts()
        return _HOSTS
