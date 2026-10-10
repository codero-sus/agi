"""OpenAI-compatible mouths: Ollama, OpenRouter, Cortex LLMHoster.

Cortex AGI stays the mind. These only speak. Keys never leave this process.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse

TIMEOUT_PROBE = 2.5
TIMEOUT_GEN = 45

OPENROUTER_HOSTS = {"openrouter.ai", "www.openrouter.ai"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _host_ok(url: str, kind: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    host = (p.hostname or "").lower()
    if not host or p.scheme not in {"http", "https"}:
        return False
    if kind == "openrouter":
        return p.scheme == "https" and host in OPENROUTER_HOSTS
    # ollama / hoster: loopback or LAN hostname, never a cloud exfil
    if host in LOCAL_HOSTS or host.endswith(".local") or host.endswith(".lan"):
        return True
    try:
        import ipaddress

        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return False


def _v1(base: str) -> str:
    b = (base or "").rstrip("/")
    return b if b.endswith("/v1") else b + "/v1"


def _headers(kind: str, key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    if kind == "openrouter":
        h["HTTP-Referer"] = os.environ.get("AGI_PUBLIC_URL", "http://localhost:8000")
        h["X-Title"] = "Cortex AGI"
    return h


def _json(url: str, kind: str, key: str = "", body: dict | None = None, timeout: float = TIMEOUT_PROBE) -> tuple[int, dict | list | str]:
    if not _host_ok(url, kind):
        raise ValueError(f"refusing {kind} url {url!r}")
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(kind, key), method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(2_000_000)
            code = getattr(resp, "status", 200)
    except urllib.error.HTTPError as e:
        raw = e.read(4000) if e.fp else b""
        try:
            return e.code, json.loads(raw.decode("utf-8", errors="replace")) if raw else {"error": str(e)}
        except json.JSONDecodeError:
            return e.code, {"error": raw.decode("utf-8", errors="replace")[:400] or str(e)}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {"error": str(e)}
    try:
        return code, json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return code, {"error": "not json", "text": raw.decode("utf-8", errors="replace")[:200]}


class RemoteChat:
    def __init__(self, kind: str, base: str, model: str = "", key: str = ""):
        self.kind = kind
        self.base = (base or "").rstrip("/")
        self.model = model or ""
        self.key = key or ""

    def generate(self, prompt: str, max_new: int = 256, temperature: float = 0.8, **_kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], max_new=max_new, temperature=temperature)

    def chat(self, messages: list[dict], max_new: int = 256, temperature: float = 0.8) -> str:
        model = self.model or self._first_model()
        if not model:
            raise RuntimeError(f"{self.kind} has no model id — pick one in Hosts")
        url = _v1(self.base) + "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max(16, int(max_new)),
            "temperature": float(temperature),
            "stream": False,
        }
        code, data = _json(url, self.kind, self.key, payload, timeout=TIMEOUT_GEN)
        if code == 404 and self.kind == "ollama":
            return self._ollama_native(messages, model, temperature)
        if code != 200 or not isinstance(data, dict):
            err = data.get("error") if isinstance(data, dict) else data
            if isinstance(err, dict):
                err = err.get("message") or err
            raise RuntimeError(f"{self.kind} {code}: {err}")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.kind} returned no choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        msg = (choice.get("message") or {}).get("content") or choice.get("text") or ""
        return str(msg).strip()

    def _ollama_native(self, messages: list[dict], model: str, temperature: float) -> str:
        code, data = _json(
            self.base + "/api/chat",
            "ollama",
            "",
            {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}},
            timeout=TIMEOUT_GEN,
        )
        if code != 200:
            raise RuntimeError(f"ollama native {code}: {data}")
        msg = (data.get("message") or {}).get("content") or data.get("response") or ""
        return str(msg).strip()

    def _first_model(self) -> str:
        models = list_models(self.kind, self.base, self.key)
        return models[0] if models else ""

    def stream(self, prompt: str, max_new: int = 256, temperature: float = 0.8):
        text = self.generate(prompt, max_new=max_new, temperature=temperature)
        yield text


def list_models(kind: str, base: str, key: str = "") -> list[str]:
    base = (base or "").rstrip("/")
    if kind == "ollama":
        code, data = _json(base + "/api/tags", "ollama", "", None, TIMEOUT_PROBE)
        if code == 200 and isinstance(data, dict):
            return [m.get("name") for m in data.get("models") or [] if m.get("name")]
        code, data = _json(_v1(base) + "/models", "ollama", "", None, TIMEOUT_PROBE)
    else:
        code, data = _json(_v1(base) + "/models", kind, key, None, TIMEOUT_PROBE)
    if code != 200 or not isinstance(data, dict):
        return []
    out = []
    for m in data.get("data") or []:
        mid = m.get("id") if isinstance(m, dict) else None
        if mid:
            out.append(mid)
    return out[:80]


def probe(kind: str, base: str, key: str = "") -> dict:
    import time

    t0 = time.perf_counter()
    try:
        models = list_models(kind, base, key)
    except Exception as e:
        return {"ok": False, "kind": kind, "error": str(e), "models": [], "ms": 0}
    ms = round((time.perf_counter() - t0) * 1000)
    if models:
        return {"ok": True, "kind": kind, "models": models, "ms": ms, "error": None}
    return {"ok": False, "kind": kind, "models": [], "ms": ms, "error": "no models (is it running?)"}
