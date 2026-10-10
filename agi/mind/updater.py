"""Self-update Cortex AGI from its GitHub origin.

The mind stays on disk (`data/`, trained weights). This only fast-forwards
source. Origin must be GitHub `codero-sus/agi` (or `AGI_UPDATE_REPO`).
No shell. Keys never logged.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from agi import __version__
from agi.config import ROOT

DEFAULT_REPO = "codero-sus/agi"
ALLOWED_HOSTS = {"github.com", "www.github.com"}
API_HOSTS = {"api.github.com"}
_LOCK = threading.RLock()
_CACHE: dict | None = None
_CACHE_AT = 0.0
CACHE_SEC = 20.0


def _repo() -> str:
    raw = (os.environ.get("AGI_UPDATE_REPO") or DEFAULT_REPO).strip()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", raw):
        return DEFAULT_REPO
    return raw


def _redact(url: str) -> str:
    try:
        p = urlsplit(url)
    except Exception:
        return ""
    host = (p.hostname or "").lower()
    if p.scheme == "ssh" or url.startswith("git@"):
        m = re.match(r"git@([^:]+):(.+)$", url)
        if m:
            return f"git@{m.group(1)}:{m.group(2)}"
        return f"{host}:{p.path.lstrip('/')}"
    netloc = host + (f":{p.port}" if p.port else "")
    return urlunsplit((p.scheme, netloc, p.path, "", ""))


def _origin_ok(url: str) -> bool:
    if not url:
        return False
    u = url.strip()
    repo = _repo().lower()
    m = re.match(r"git@([^:]+):(.+)$", u)
    if m:
        host, path = m.group(1).lower(), m.group(2).lower().removesuffix(".git")
        return host in ALLOWED_HOSTS and path == repo
    try:
        p = urlparse(u)
    except Exception:
        return False
    host = (p.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        return False
    path = (p.path or "").lower().strip("/").removesuffix(".git")
    return path == repo


def _git(*args: str, timeout: float = 25) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        p = subprocess.run(  # noqa: S603
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", "git not installed"
    except subprocess.TimeoutExpired:
        return 124, "", "git timed out"
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def _git_one(*args: str, timeout: float = 10) -> str:
    code, out, _err = _git(*args, timeout=timeout)
    return out if code == 0 else ""


def local() -> dict:
    sha = _git_one("rev-parse", "HEAD")
    branch = _git_one("rev-parse", "--abbrev-ref", "HEAD") or "HEAD"
    origin = _git_one("remote", "get-url", "origin")
    porcelain = _git_one("status", "--porcelain")
    dirty = []
    for ln in porcelain.splitlines():
        if not ln.strip():
            continue
        path = ln[3:] if len(ln) > 3 and ln[2] == " " else ln[2:].lstrip()
        if " -> " in path:
            path = path.split(" -> ", 1)[-1]
        dirty.append(path)
    dirty = dirty[:24]
    return {
        "version": __version__,
        "sha": sha[:12] if sha else "",
        "sha_full": sha,
        "branch": branch,
        "origin": _redact(origin),
        "origin_ok": _origin_ok(origin),
        "dirty": dirty,
        "git": bool(sha),
    }


def _github_head(branch: str) -> dict:
    repo = _repo()
    url = f"https://api.github.com/repos/{repo}/commits/{branch}"
    p = urlparse(url)
    if (p.hostname or "").lower() not in API_HOSTS or p.scheme != "https":
        return {"ok": False, "error": "refusing api host"}
    req = Request(
        url,
        headers={
            "User-Agent": f"Cortex-AGI/{__version__}",
            "Accept": "application/vnd.github+json",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=8) as resp:  # noqa: S310
            raw = resp.read(200_000)
    except HTTPError as e:
        return {"ok": False, "error": f"github {e.code}"}
    except (URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": str(e)}
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"ok": False, "error": "github not json"}
    sha = data.get("sha") or ""
    msg = ((data.get("commit") or {}).get("message") or "").split("\n", 1)[0][:160]
    return {"ok": bool(sha), "sha": sha, "message": msg, "error": None if sha else "no sha"}


def status() -> dict:
    """Local SHA only — no network. Uses cache if a check already ran."""
    if _CACHE is not None and time.time() - _CACHE_AT < CACHE_SEC * 6:
        return _CACHE
    loc = local()
    return {
        **loc,
        "ok": True,
        "available": False,
        "behind": 0,
        "ahead": 0,
        "remote_sha": "",
        "remote_message": "",
        "error": None,
        "checked": False,
    }


def check(*, force: bool = False) -> dict:
    global _CACHE, _CACHE_AT
    now = time.time()
    if not force and _CACHE is not None and now - _CACHE_AT < CACHE_SEC:
        return _CACHE
    with _LOCK:
        loc = local()
        out = {**loc, "ok": True, "available": False, "behind": 0, "ahead": 0, "remote_sha": "", "remote_message": "", "error": None, "checked": True}
        if not loc["git"]:
            out["ok"] = False
            out["error"] = "not a git checkout"
            _CACHE, _CACHE_AT = out, now
            return out
        if loc["branch"] in {"HEAD", ""}:
            out["ok"] = False
            out["error"] = "detached HEAD — checkout a branch to update"
            _CACHE, _CACHE_AT = out, now
            return out
        if not loc["origin_ok"]:
            out["ok"] = False
            out["error"] = f"origin is not github.com/{_repo()} — refusing"
            _CACHE, _CACHE_AT = out, now
            return out
        code, _o, err = _git("fetch", "--depth", "50", "origin", loc["branch"], timeout=40)
        remote_sha = _git_one("rev-parse", f"origin/{loc['branch']}")
        if not remote_sha:
            gh = _github_head(loc["branch"])
            if gh.get("ok"):
                remote_sha = gh["sha"]
                out["remote_message"] = gh.get("message") or ""
            elif code != 0:
                out["ok"] = False
                out["error"] = "fetch failed (offline?)"
                _CACHE, _CACHE_AT = out, now
                return out
        else:
            out["remote_message"] = _git_one("log", "-1", "--format=%s", f"origin/{loc['branch']}")[:160]
        out["remote_sha"] = (remote_sha or "")[:12]
        if remote_sha:
            behind = _git_one("rev-list", "--count", f"HEAD..origin/{loc['branch']}")
            ahead = _git_one("rev-list", "--count", f"origin/{loc['branch']}..HEAD")
            try:
                out["behind"] = int(behind or 0)
                out["ahead"] = int(ahead or 0)
            except ValueError:
                out["behind"] = 0 if loc["sha_full"] == remote_sha else 1
                out["ahead"] = 0
            if out["behind"] == 0 and out["ahead"] == 0 and loc["sha_full"] != remote_sha:
                # shallow clone may not know ancestry — different SHA still means we can pull
                out["behind"] = 1
        out["available"] = out["behind"] > 0 and out["ahead"] == 0 and not loc["dirty"]
        if loc["dirty"]:
            out["error"] = "working tree dirty — commit or stash before applying"
        elif out["ahead"] and out["behind"]:
            out["error"] = "diverged from origin — refusing to apply"
            out["available"] = False
        elif out["ahead"] and not out["behind"]:
            out["error"] = None
        if err and "couldn't find remote ref" in err.lower():
            out["error"] = f"origin has no {loc['branch']}"
            out["available"] = False
        _CACHE, _CACHE_AT = out, time.time()
        return out


def _pip() -> str:
    try:
        p = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "-q", "--disable-pip-version-check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return str(e)
    if p.returncode != 0:
        return (p.stderr or p.stdout or "pip failed")[:300]
    return ""


def _restart_later() -> None:
    def _go():
        time.sleep(1.2)
        os.chdir(str(ROOT))
        os.execv(sys.executable, [sys.executable, "-m", "agi"])

    threading.Thread(target=_go, daemon=True, name="cortex-restart").start()


def apply(*, restart: bool = True) -> dict:
    with _LOCK:
        info = check(force=True)
        if not info.get("ok"):
            return {**info, "applied": False}
        if info.get("dirty"):
            return {**info, "applied": False, "error": "working tree dirty — commit or stash before applying"}
        if info.get("ahead") and info.get("behind"):
            return {**info, "applied": False, "error": "diverged from origin"}
        if not info.get("behind"):
            return {**info, "applied": False, "error": None, "message": "already current"}
        req_before = ""
        req_path = ROOT / "requirements.txt"
        if req_path.exists():
            req_before = req_path.read_text(encoding="utf-8")
        branch = info["branch"]
        code, out, err = _git("merge", "--ff-only", f"origin/{branch}", timeout=40)
        if code != 0:
            return {**info, "applied": False, "error": (err or out or "merge failed")[:300]}
        pip_err = ""
        if req_path.exists() and req_path.read_text(encoding="utf-8") != req_before:
            pip_err = _pip()
        global _CACHE, _CACHE_AT
        _CACHE, _CACHE_AT = None, 0.0
        after = local()
        pulled = int(info.get("behind") or 0)
        result = {
            **info,
            **after,
            "ok": True,
            "applied": True,
            "available": False,
            "behind": 0,
            "pulled": pulled,
            "message": out.splitlines()[-1] if out else "fast-forwarded",
            "pip_error": pip_err or None,
            "restarting": False,
            "error": None,
        }
        want = restart and os.environ.get("AGI_UPDATE_RESTART", "1") not in {"0", "false", "no"}
        if want:
            result["restarting"] = True
            _restart_later()
        return result
