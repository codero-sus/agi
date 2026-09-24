"""Custom AGI server.

Serves the control-plane UI, a WebSocket thought stream, a native AGI API,
and an OpenAI-compatible `/v1/chat/completions` endpoint — all backed by the
local GGUF / safetensors loader. Nothing is proxied to a vendor.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agi import __version__
from agi.config import HOST, PORT, WEB_DIR, ensure_dirs
from agi.mind.core import get_agi

ensure_dirs()

app = FastAPI(title="CORTEX", version=__version__, description="Self-improving AGI server")
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


class CompletionsIn(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float | None = 0.8
    max_tokens: int | None = 256
    stream: bool = False


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "name": "CORTEX", "version": __version__}


@app.get("/api/state")
def api_state():
    return get_agi().state()


@app.post("/api/chat")
def api_chat(body: ChatIn):
    return get_agi().chat(body.message)


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
            msg = (data.get("message") or data.get("text") or "").strip()
            if data.get("type") == "state" or not msg:
                await ws.send_json({"type": "state", "state": agi.state()})
                continue
            if data.get("type") == "improve":
                events = agi.improver.cycle(reason="ws")
                await ws.send_json({"type": "improve", "events": events})
                await ws.send_json({"type": "state", "state": agi.state()})
                continue
            for event in agi.think(msg):
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
def v1_chat(body: CompletionsIn):
    agi = get_agi()
    user = ""
    for m in reversed(body.messages):
        if m.get("role") == "user":
            user = str(m.get("content") or "")
            break
    if not user:
        user = str(body.messages[-1].get("content") if body.messages else "")

    if body.stream:
        def gen():
            full = []
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            for event in agi.think(user):
                if event.get("type") == "token":
                    piece = event.get("text") or ""
                    full.append(piece)
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

    result = agi.chat(user)
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
        "agi": {"intent": result.get("intent"), "thoughts": result.get("thoughts")},
    }


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
