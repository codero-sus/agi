"""Built-in tools the cognitive loop can call."""

from __future__ import annotations

import ast
import datetime as dt
import math
import traceback
from typing import Any, Callable


def run_python(code: str, timeout_hint: int = 1) -> str:
    """Execute a small Python snippet in a restricted namespace."""
    code = code.strip()
    if not code:
        return "(empty)"
    if len(code) > 4000:
        return "Refusing: code too long."
    forbidden = ("import os", "import sys", "subprocess", "socket", "shutil", "open(", "exec(", "eval(", "__")
    low = code.lower()
    if any(f in low for f in ("import os", "import sys", "subprocess", "socket", "shutil")):
        return "Refusing: that import is not allowed in the sandbox."
    ns: dict[str, Any] = {
        "math": math,
        "datetime": dt,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "print": print,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "enumerate": enumerate,
        "zip": zip,
        "round": round,
    }
    buf: list[str] = []

    def _print(*args, **kwargs):
        buf.append(" ".join(str(a) for a in args))

    ns["print"] = _print
    try:
        tree = ast.parse(code, mode="exec")
        last = tree.body[-1] if tree.body else None
        if isinstance(last, ast.Expr):
            exec(compile(ast.Module(tree.body[:-1], []), "<skill>", "exec"), ns, ns)
            val = eval(compile(ast.Expression(last.value), "<skill>", "eval"), ns, ns)
            if val is not None:
                buf.append(repr(val))
        else:
            exec(compile(tree, "<skill>", "exec"), ns, ns)
    except Exception:
        buf.append(traceback.format_exc(limit=2))
    return "\n".join(buf) if buf else "(ok)"


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable[..., str]] = {}
        self._desc: dict[str, str] = {}
        self.register("python", "Run a short Python snippet.", self._python)
        self.register("now", "Current UTC date and time.", self._now)
        self.register("math", "Evaluate an arithmetic expression.", self._math)

    def register(self, name: str, description: str, fn: Callable[..., str]) -> None:
        self._tools[name] = fn
        self._desc[name] = description

    def names(self) -> list[dict]:
        return [{"name": k, "description": self._desc.get(k, "")} for k in self._tools]

    def call(self, name: str, **kwargs) -> str:
        fn = self._tools.get(name)
        if not fn:
            return f"unknown tool {name}"
        return fn(**kwargs)

    def _python(self, code: str = "") -> str:
        return run_python(code)

    def _now(self, **_kwargs) -> str:
        return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _math(self, expr: str = "") -> str:
        from agi.mind.knowledge import try_math

        v = try_math(expr)
        return v if v is not None else f"could not evaluate: {expr}"
