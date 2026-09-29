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

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from agi import __version__
from agi.config import HOST, PORT, WEB_DIR, ensure_dirs
from agi.mind.agent import TOOL_NAMES, get_roster
from agi.mind.core import get_agi
from agi.mind.desk import get_desk
from agi.mind.workspace import PROMPTS, get_workspace


class CacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=5"
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


class SessionIn(BaseModel):
    title: str | None = None
    pinned: bool | None = None


class DocIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    body: str = Field("", max_length=80000)
    kind: str = "note"


class DocPatch(BaseModel):
    title: str | None = None
    body: str | None = None


class TaskIn(BaseModel):
    title: str | None = None
    done: bool | None = None


class ResearchIn(BaseModel):
    topic: str = Field(..., min_length=2, max_length=400)


class AgentIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=40)
    mission: str = Field("", max_length=400)
    tools: list[str] = Field(default_factory=list)


class AgentRunIn(BaseModel):
    goal: str = Field(..., min_length=1, max_length=400)
    agent: str | None = None


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


@app.get("/api/prompts")
def api_prompts():
    return {"prompts": PROMPTS}


@app.get("/api/sessions")
def api_sessions():
    return {"sessions": get_workspace().list()}


@app.post("/api/sessions")
def api_session_new(body: SessionIn | None = None):
    title = (body.title if body else None) or "new thread"
    return get_workspace().create(title)


@app.get("/api/sessions/{sid}")
def api_session_get(sid: str):
    s = get_workspace().get(sid)
    if not s:
        return JSONResponse({"error": "missing"}, status_code=404)
    return s


@app.patch("/api/sessions/{sid}")
def api_session_patch(sid: str, body: SessionIn):
    s = get_workspace().patch(sid, title=body.title, pinned=body.pinned)
    if not s:
        return JSONResponse({"error": "missing"}, status_code=404)
    return {"ok": True, "id": sid, "title": s.get("title"), "pinned": s.get("pinned")}


@app.delete("/api/sessions/{sid}")
def api_session_del(sid: str):
    ok = get_workspace().delete(sid)
    return {"ok": ok}


@app.get("/api/vault")
def api_vault():
    return {"files": get_workspace().vault()}


@app.get("/api/vault/{name}")
def api_vault_read(name: str):
    text = get_workspace().read_vault(name)
    if text is None:
        return JSONResponse({"error": "missing"}, status_code=404)
    return {"name": name, "text": text}


@app.post("/api/ingest")
async def api_ingest(file: UploadFile = File(...)):
    raw = await file.read()
    result = get_workspace().ingest(file.filename or "upload.txt", raw)
    if result.get("ok"):
        stem = (file.filename or "upload").rsplit("/", 1)[-1].rsplit(".", 1)[0][:80]
        get_agi().teach(stem or "upload", (result.get("excerpt") or "")[:4000])
    return result


@app.get("/api/docs")
def api_docs():
    return {"docs": get_desk().list_docs()}


@app.post("/api/docs")
def api_doc_new(body: DocIn):
    return get_desk().write_doc(body.title, body.body, kind=body.kind)


@app.get("/api/docs/{did}")
def api_doc_get(did: str):
    d = get_desk().get_doc(did)
    if not d:
        return JSONResponse({"error": "missing"}, status_code=404)
    return d


@app.patch("/api/docs/{did}")
def api_doc_patch(did: str, body: DocPatch):
    d = get_desk().save_doc(did, title=body.title, body=body.body)
    if not d:
        return JSONResponse({"error": "missing"}, status_code=404)
    return d


@app.delete("/api/docs/{did}")
def api_doc_del(did: str):
    return {"ok": get_desk().delete_doc(did)}


@app.get("/api/tasks")
def api_tasks():
    return {"tasks": get_desk().list_tasks()}


@app.post("/api/tasks")
def api_task_new(body: TaskIn):
    if not (body.title or "").strip():
        return JSONResponse({"error": "title required"}, status_code=400)
    return get_desk().add_task(body.title.strip(), source="api")


@app.patch("/api/tasks/{tid}")
def api_task_patch(tid: str, body: TaskIn):
    t = get_desk().patch_task(tid, done=body.done, title=body.title if body.title else None)
    if not t:
        return JSONResponse({"error": "missing"}, status_code=404)
    return t


@app.delete("/api/tasks/{tid}")
def api_task_del(tid: str):
    return {"ok": get_desk().delete_task(tid)}


@app.post("/api/research")
def api_research(body: ResearchIn):
    from agi.mind.research import investigate

    report = investigate(get_agi(), body.topic)
    return {
        "title": report.title,
        "markdown": report.markdown,
        "confidence": report.confidence,
        "doc_id": report.doc_id,
        "sources": [{"kind": s.kind, "title": s.title, "url": s.url} for s in report.sources],
        "chain": report.chain,
        "tasks": report.tasks,
    }


@app.post("/api/compare")
def api_compare(body: ResearchIn):
    from agi.mind.research import compare_brief

    report = compare_brief(get_agi(), body.topic)
    return {
        "title": report.title,
        "markdown": report.markdown,
        "confidence": report.confidence,
        "doc_id": report.doc_id,
        "chain": report.chain,
    }


@app.post("/api/agent")
def api_agent(body: ResearchIn):
    from agi.mind.agent import act

    run = act(get_agi(), body.topic)
    return {
        "goal": run.goal,
        "markdown": run.markdown,
        "confidence": run.confidence,
        "doc_id": run.doc_id,
        "tasks": run.tasks,
        "agent": run.agent,
    }


@app.get("/api/agents")
def api_agents():
    return {"agents": get_roster().list(), "tools": list(TOOL_NAMES)}


@app.post("/api/agents")
def api_agent_new(body: AgentIn):
    try:
        spec = get_roster().create(body.name, body.mission, body.tools, created_by="api")
        return spec.as_dict()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/agents/{key}")
def api_agent_get(key: str):
    spec = get_roster().get(key)
    if not spec:
        return JSONResponse({"error": "missing"}, status_code=404)
    return spec.as_dict()


@app.delete("/api/agents/{key}")
def api_agent_del(key: str):
    return {"ok": get_roster().delete(key)}


@app.post("/api/agents/{key}/run")
def api_agent_run(key: str, body: AgentRunIn):
    from agi.mind.agent import act

    run = act(get_agi(), f"run {key}: {body.goal}")
    return {
        "agent": run.agent,
        "goal": run.goal,
        "markdown": run.markdown,
        "confidence": run.confidence,
        "doc_id": run.doc_id,
        "tasks": run.tasks,
    }


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
            sid = data.get("session_id")
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
            if sid:
                get_workspace().append(sid, "user", msg)
            reply = ""
            chain = None
            async for event in iter_think(msg):
                if event.get("type") == "done":
                    reply = event.get("message") or ""
                    chain = event.get("chain")
                await ws.send_json(event)
            if sid and reply:
                get_workspace().append(sid, "agi", reply, meta={"chain": chain} if chain else None)
            await ws.send_json({"type": "state", "state": agi.state()})
            if sid:
                await ws.send_json({"type": "session", "session": get_workspace().get(sid)})
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
