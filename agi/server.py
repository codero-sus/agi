"""Custom AGI server.

Serves the control-plane UI, a WebSocket thought stream, a native AGI API,
and an OpenAI-compatible `/v1/chat/completions` endpoint — all backed by the
local GGUF / safetensors loader. Nothing is proxied to a vendor.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from agi import __version__
from agi.config import HOST, PORT, WEB_DIR, ensure_dirs
from agi.mind.core import get_agi


class CacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=120"
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_dirs()
    get_agi()
    yield
    try:
        get_agi().engine.trainer.flush_save()
    except Exception:
        pass


ensure_dirs()

app = FastAPI(
    title="CORTEX",
    version=__version__,
    description="Self-improving AGI server",
    lifespan=lifespan,
)
app.add_middleware(CacheStaticMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=400)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class TeachIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    body: str = Field(..., min_length=1, max_length=4000)


class CompletionsIn(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float | None = 0.8
    max_tokens: int | None = 256
    stream: bool = False


async def iter_think(message: str) -> AsyncIterator[dict]:
    """Run the blocking cognitive loop in a thread; yield events as they happen."""
    loop = asyncio.get_running_loop()
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            for event in get_agi().think(message):
                q.put(event)
        except Exception as exc:
            q.put({"type": "error", "text": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True, name="cortex-think").start()
    while True:
        event = await loop.run_in_executor(None, q.get)
        if event is None:
            break
        yield event


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "name": "CORTEX", "version": __version__}


@app.get("/api/state")
def api_state():
    return get_agi().state()


@app.get("/api/history")
def api_history(k: int = Query(40, ge=1, le=200)):
    return {"messages": get_agi().memory.history(k=k)}


@app.get("/api/search")
def api_search(q: str = Query(..., min_length=1, max_length=400)):
    agi = get_agi()
    return {
        "query": q,
        "episodes": [
            {"id": e.id, "role": e.role, "content": e.content, "score": e.score}
            for e in agi.memory.search(q, k=8)
        ],
        "facts": [
            {"subject": f.subject, "predicate": f.predicate, "object": f.obj, "confidence": f.confidence}
            for f in agi.memory.facts_about(q, k=8)
        ],
        "knowledge": [{"title": a.title, "body": a.body} for a in agi.knowledge.search(q, k=4)],
    }


@app.post("/api/chat")
def api_chat(body: ChatIn):
    return get_agi().chat(body.message)


@app.post("/api/teach")
def api_teach(body: TeachIn):
    return get_agi().teach(body.title, body.body)


@app.post("/api/forget")
def api_forget(q: str = Query(..., min_length=1, max_length=200)):
    n = get_agi().memory.forget_facts(q)
    return {"forgotten": n, "query": q}


@app.post("/api/stop")
def api_stop():
    get_agi().stop()
    return {"ok": True}


@app.get("/api/export")
def api_export():
    agi = get_agi()
    lines = []
    for m in agi.memory.history(k=200):
        lines.append(f"{m['role']}: {m['content']}")
    text = "\n\n".join(lines) or "(empty mind)"
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


@app.post("/api/improve")
def api_improve():
    agi = get_agi()
    events = agi.improver.cycle(reason="api")
    return {"events": events, "state": agi.state()}


@app.post("/api/reload-model")
def api_reload():
    agi = get_agi()
    info = agi.engine.reload()
    return info.as_dict()


@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    agi = get_agi()
    try:
        await ws.send_json({"type": "hello", "state": agi.state()})
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"type": "chat", "message": raw}
            kind = data.get("type") or "chat"
            msg = (data.get("message") or data.get("text") or "").strip()
            if kind == "ping":
                await ws.send_json({"type": "pong", "t": time.time()})
                continue
            if kind == "stop":
                agi.stop()
                await ws.send_json({"type": "stopped"})
                continue
            if kind == "state" or (kind == "chat" and not msg):
                await ws.send_json({"type": "state", "state": agi.state()})
                continue
            if kind == "improve":
                events = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: agi.improver.cycle(reason="ws")
                )
                await ws.send_json({"type": "improve", "events": events})
                await ws.send_json({"type": "state", "state": agi.state()})
                continue
            async for event in iter_think(msg):
                await ws.send_json(event)
            await ws.send_json({"type": "state", "state": agi.state()})
    except WebSocketDisconnect:
        return


@app.get("/v1/models")
def v1_models():
    agi = get_agi()
    info = agi.engine.info
    return {
        "object": "list",
        "data": [
            {
                "id": f"cortex-{info.source}",
                "object": "model",
                "owned_by": "local-agi",
                "backend": info.backend,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def v1_chat(body: CompletionsIn):
    agi = get_agi()
    user = ""
    for m in reversed(body.messages):
        if m.get("role") == "user":
            user = str(m.get("content") or "")
            break
    if not user:
        user = str(body.messages[-1].get("content") if body.messages else "")

    if body.stream:
        async def gen():
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            async for event in iter_think(user):
                if event.get("type") == "token":
                    piece = event.get("text") or ""
                    payload = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            done = {
                "id": cid,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    result = await asyncio.get_running_loop().run_in_executor(None, agi.chat, user)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"cortex-{agi.engine.info.source}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["message"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "agi": {
            "intent": result.get("intent"),
            "thoughts": result.get("thoughts"),
            "latency_ms": result.get("latency_ms"),
        },
    }


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
