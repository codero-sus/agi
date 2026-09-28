"""Safe outbound reads — Wikipedia and public HTTP text, never local/private nets."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "CORTEX-AGI/0.5 (local research; +https://github.com/codero-sus/agi)"
TIMEOUT = 8
MAX_BYTES = 180_000
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        t = data.strip()
        if t:
            self.parts.append(t)


def _private(host: str) -> bool:
    host = (host or "").strip().lower().rstrip(".")
    if not host or host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True
    if host.endswith(".internal") or host.endswith(".localhost"):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return True
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            return True
    return False


def allowed_url(url: str) -> bool:
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return False
    if p.scheme not in {"http", "https"}:
        return False
    host = p.hostname or ""
    if not host or _private(host):
        return False
    return True


def _get(url: str) -> bytes | None:
    if not allowed_url(url):
        return None
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, text/plain, text/html"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310
            return resp.read(MAX_BYTES)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def wiki_search(query: str, k: int = 5) -> list[dict]:
    q = (query or "").strip()[:180]
    if not q:
        return []
    params = urllib.parse.urlencode(
        {"action": "opensearch", "search": q, "limit": k, "namespace": 0, "format": "json"}
    )
    raw = _get(f"{WIKI_API}?{params}")
    if not raw:
        return []
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    titles = data[1] if len(data) > 1 else []
    descs = data[2] if len(data) > 2 else []
    urls = data[3] if len(data) > 3 else []
    out = []
    for i, title in enumerate(titles):
        out.append(
            {
                "title": title,
                "description": descs[i] if i < len(descs) else "",
                "url": urls[i] if i < len(urls) else "",
            }
        )
    return out


def wiki_summary(title: str) -> dict | None:
    slug = urllib.parse.quote(title.replace(" ", "_"), safe="")
    raw = _get(WIKI_SUMMARY + slug)
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    return {
        "title": data.get("title") or title,
        "extract": extract[:4000],
        "url": (data.get("content_urls") or {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{slug}",
        "description": data.get("description") or "",
    }


def fetch_text(url: str) -> str | None:
    raw = _get(url)
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    if "<" in text[:400].lower():
        p = _TextExtractor()
        try:
            p.feed(text)
            p.close()
        except Exception:
            return re.sub(r"<[^>]+>", " ", text)[:8000]
        return re.sub(r"\n{3,}", "\n\n", " ".join(p.parts))[:8000]
    return text[:8000]
