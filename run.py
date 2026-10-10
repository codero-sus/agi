#!/usr/bin/env python3
"""Launch Cortex AGI with the interpreter in python.env (project root)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _from_python_env() -> str | None:
    p = ROOT / "python.env"
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("python="):
            line = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not line:
            continue
        path = Path(line)
        if not path.is_absolute():
            path = ROOT / path
        return str(path)
    return None


if __name__ == "__main__":
    wanted = (os.environ.get("AGI_PYTHON") or "").strip() or _from_python_env()
    if wanted:
        try:
            same = Path(wanted).resolve() == Path(sys.executable).resolve()
        except OSError:
            same = False
        if not same:
            os.execv(wanted, [wanted, str(ROOT / "run.py"), *sys.argv[1:]])
    from agi.server import run

    run()
