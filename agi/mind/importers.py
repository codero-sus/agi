"""Import chat history from WhatsApp, ChatGPT, Claude, Telegram, and cousins.

The file becomes episodes, a thread, a document, and background training.
Not a dump into a context window — the mind actually keeps it.
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agi.mind.desk import get_desk
from agi.mind.workspace import get_workspace

if TYPE_CHECKING:
    from agi.mind.core import AGI

MAX_BYTES = 25_000_000
MAX_MSGS = 5000
MAX_TRAIN = 48
MAX_THREAD = 240
SKIP_WA = re.compile(
    r"(media omitted|omitted>|messages and calls are end-to-end|"
    r"this message was deleted|you deleted this message|"
    r"waiting for this message|missed (voice|video) call)",
    re.I,
)

# WhatsApp: 12/31/23, 10:15 PM - Name: text
# WhatsApp: [31/12/2023, 22:15:12] Name: text
WA_LINE = re.compile(
    r"^\[?(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4}),?\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap]\.?\s*[Mm]\.?)?)\]?\s*-?\s*"
    r"(.+?):\s(.*)$"
)

ASSISTANT_ROLES = {
    "assistant",
    "agi",
    "chatgpt",
    "gpt",
    "claude",
    "gemini",
    "bard",
    "bot",
    "model",
    "system",
    "cortex",
}
USER_ROLES = {"user", "human", "me", "you", "customer", "prompter"}

TRANSFER_PROMPT = """You are exporting this conversation so CORTEX, a local AGI running on my machine, can remember it and train on it.

Reply with JSON only. No markdown fences. No commentary before or after.

Schema:
{
  "cortex_export": 1,
  "source": "chatgpt",
  "threads": [
    {
      "title": "short title for this chat",
      "turns": [
        {"role": "user", "content": "what I said"},
        {"role": "agi", "content": "what you said"}
      ]
    }
  ]
}

Rules:
- role is only "user" or "agi" (you are agi)
- Include the real conversation we have had, in order. Do not invent turns.
- If it is long: first 20 turns + most recent 200 turns.
- Plain text only. No HTML, no tool dumps, no chain-of-thought hidden traces.
- source is chatgpt, claude, gemini, grok, copilot, or other.
"""


@dataclass
class Turn:
    role: str
    name: str
    content: str
    ts: float | None = None


@dataclass
class Thread:
    title: str
    source: str
    turns: list[Turn] = field(default_factory=list)


def absorb(agi: "AGI", filename: str, data: bytes) -> dict:
    if len(data) > MAX_BYTES:
        return {"ok": False, "error": "file too large (25MB cap)."}
    try:
        threads = parse(filename, data)
    except Exception as e:
        return {"ok": False, "error": f"could not parse: {e}"}
    if not threads:
        return {
            "ok": False,
            "error": "no chat turns found. export WhatsApp as .txt/.zip, ChatGPT conversations.json, Claude JSON, Telegram result.json, or messages JSONL.",
        }

    n_turns = sum(len(t.turns) for t in threads)
    stored = 0
    trained = 0
    session_ids: list[str] = []
    items: list[tuple[str, str]] = []
    for th in threads:
        for turn in th.turns[:MAX_MSGS]:
            text = turn.content.strip()
            if len(text) < 2:
                continue
            role = "agi" if turn.role == "agi" else "user"
            label = f"{turn.name}: {text}" if turn.name and turn.role != "agi" else text
            items.append((role, label[:2000]))
            if len(items) >= MAX_MSGS:
                break
        if len(items) >= MAX_MSGS:
            break
    stored = agi.memory.remember_many(items)

    ws = get_workspace()
    for th in threads[:40]:
        title = f"{th.source}: {th.title}"[:80]
        s = ws.create(title)
        ws.append_many(s["id"], [(t.role if t.role in ("user", "agi") else "user", t.content[:2000]) for t in th.turns[-MAX_THREAD:]])
        session_ids.append(s["id"])

    snips = _train_snips(threads)
    for snip in snips[:MAX_TRAIN]:
        agi.engine.train_on(snip, steps=2, blocking=False)
        trained += 1

    sources = sorted({t.source for t in threads})
    summary = (
        f"# Imported {filename}\n\n"
        f"**Source** {', '.join(sources)}  \n"
        f"**Threads** {len(threads)} · **turns** {n_turns} · **remembered** {stored} · **train queue** {min(trained, MAX_TRAIN)}\n\n"
        "## Threads\n"
        + "\n".join(f"- {th.source}: {th.title} ({len(th.turns)} turns)" for th in threads[:30])
        + "\n\nThe neural core is training on these dialogues in the background.\n"
    )
    doc = get_desk().write_doc(f"Import: {filename}"[:80], summary, kind="import")
    agi.teach(f"Imported {sources[0]} chats", summary[:2400])
    agi.memory.add_fact("import", "from", ",".join(sources)[:80], 0.9)
    return {
        "ok": True,
        "filename": filename,
        "source": sources,
        "threads": len(threads),
        "turns": n_turns,
        "remembered": stored,
        "trained": min(len(snips), MAX_TRAIN),
        "doc_id": doc["id"],
        "sessions": session_ids[:8],
    }


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse(filename: str, data: bytes) -> list[Thread]:
    name = (filename or "export").lower()
    if name.endswith(".zip") or data[:2] == b"PK":
        return _from_zip(data)
    text = _decode(data)
    if name.endswith(".jsonl") or (text.lstrip().startswith("{") and "\n{" in text[:2000]):
        got = _from_jsonl(text)
        if got:
            return got
    if name.endswith(".json") or text.lstrip()[:1] in "[{":
        got = _from_json(text)
        if got:
            return got
    if name.endswith(".csv"):
        got = _from_csv(text)
        if got:
            return got
    got = _from_whatsapp(text, filename)
    if got:
        return got
    got = _from_markdown(text, filename)
    return got


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _from_zip(data: bytes) -> list[Thread]:
    out: list[Thread] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return []
    for info in zf.infolist():
        if info.is_dir() or info.file_size > MAX_BYTES:
            continue
        base = info.filename.replace("\\", "/").split("/")[-1].lower()
        if not base.endswith((".json", ".jsonl", ".txt", ".csv")):
            continue
        if info.file_size > 8_000_000 and not base.endswith(".json"):
            continue
        try:
            raw = zf.read(info)
        except Exception:
            continue
        out.extend(parse(base, raw))
        if sum(len(t.turns) for t in out) >= MAX_MSGS:
            break
    return out


def extract_json(text: str) -> str:
    t = strip_fences(text)
    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        return t[i : j + 1]
    i, j = t.find("["), t.rfind("]")
    if i >= 0 and j > i:
        return t[i : j + 1]
    return t


def _from_json(text: str) -> list[Thread]:
    try:
        data = json.loads(extract_json(text))
    except json.JSONDecodeError:
        return []
    return _from_obj(data)


def _from_cortex(data: dict) -> list[Thread]:
    src = str(data.get("source") or "cortex")[:40]
    raw_threads = data.get("threads")
    out: list[Thread] = []
    if isinstance(raw_threads, list):
        for th in raw_threads:
            if not isinstance(th, dict):
                continue
            title = str(th.get("title") or src)[:80]
            turns = _messages_of(th.get("turns") or th.get("messages") or [])
            if turns:
                out.append(Thread(title, src, turns))
    if not out and isinstance(data.get("turns"), list):
        turns = _messages_of(data["turns"])
        if turns:
            out.append(Thread(str(data.get("title") or src)[:80], src, turns))
    return out


def _from_obj(data: Any) -> list[Thread]:
    if isinstance(data, dict) and (
        data.get("cortex_export") or data.get("cortex") or data.get("format") == "cortex"
    ):
        got = _from_cortex(data)
        if got:
            return got
    if isinstance(data, list):
        if not data:
            return []
        if all(isinstance(x, dict) and ("mapping" in x or "title" in x) for x in data[:3]):
            threads = []
            for conv in data:
                th = _chatgpt_one(conv)
                if th and th.turns:
                    threads.append(th)
            if threads:
                return threads
        if all(isinstance(x, dict) and ("chat_messages" in x or "name" in x) for x in data[:3]):
            threads = []
            for conv in data:
                th = _claude_one(conv)
                if th and th.turns:
                    threads.append(th)
            if threads:
                return threads
        msgs = _messages_of(data)
        if msgs:
            return [Thread("imported", "json", msgs)]
        return []
    if not isinstance(data, dict):
        return []
    if "mapping" in data or (data.get("title") and "mapping" in data):
        th = _chatgpt_one(data)
        return [th] if th and th.turns else []
    if isinstance(data.get("conversations"), list):
        nested = _from_obj(data["conversations"])
        if nested:
            return nested
    if "chats" in data:  # Telegram
        return _telegram(data)
    if isinstance(data.get("messages"), list):
        msgs = _messages_of(data["messages"])
        if msgs:
            title = str(data.get("title") or data.get("name") or "chat")
            src = "telegram" if data.get("type") or data.get("id") else "openai"
            return [Thread(title[:80], src, msgs)]
    if isinstance(data.get("chat_messages"), list):
        th = _claude_one(data)
        return [th] if th and th.turns else []
    msgs = _messages_of(data)
    return [Thread("imported", "json", msgs)] if msgs else []


def _chatgpt_one(conv: dict) -> Thread | None:
    title = str(conv.get("title") or "ChatGPT")[:80]
    mapping = conv.get("mapping")
    turns: list[Turn] = []
    if isinstance(mapping, dict):
        node_id = conv.get("current_node")
        if not node_id:
            for k, v in mapping.items():
                if isinstance(v, dict) and not v.get("parent"):
                    node_id = k
                    break
        order: list[str] = []
        seen = set()
        # walk back to root then reverse
        cur = node_id
        while cur and cur not in seen and cur in mapping:
            seen.add(cur)
            order.append(cur)
            parent = (mapping.get(cur) or {}).get("parent")
            cur = parent
        order.reverse()
        if len(order) < 2:
            # walk forward from roots
            order = []
            roots = [k for k, v in mapping.items() if isinstance(v, dict) and not v.get("parent")]
            stack = list(roots)
            seen = set()
            while stack:
                nid = stack.pop(0)
                if nid in seen:
                    continue
                seen.add(nid)
                order.append(nid)
                kids = (mapping.get(nid) or {}).get("children") or []
                stack.extend(kids)
        for nid in order:
            node = mapping.get(nid) or {}
            msg = node.get("message") or {}
            turn = _turn_from_gpt_message(msg)
            if turn:
                turns.append(turn)
    if not turns and isinstance(conv.get("messages"), list):
        turns = _messages_of(conv["messages"])
    if not turns:
        return None
    return Thread(title, "chatgpt", turns)


def _turn_from_gpt_message(msg: dict) -> Turn | None:
    if not isinstance(msg, dict):
        return None
    author = msg.get("author") or {}
    role = str(author.get("role") or msg.get("role") or "")
    content = msg.get("content")
    text = ""
    if isinstance(content, dict):
        parts = content.get("parts") or []
        text = "\n".join(p if isinstance(p, str) else json.dumps(p)[:400] for p in parts if p)
    elif isinstance(content, str):
        text = content
    text = (text or "").strip()
    if not text or content and isinstance(content, dict) and content.get("content_type") not in (None, "text", "multimodal_text"):
        if not text:
            return None
    mapped = _role(role, "")
    return Turn(mapped, role or mapped, text[:4000], msg.get("create_time"))


def _claude_one(conv: dict) -> Thread | None:
    title = str(conv.get("name") or conv.get("title") or "Claude")[:80]
    raw = conv.get("chat_messages") or conv.get("messages") or []
    turns = _messages_of(raw)
    if not turns:
        return None
    return Thread(title, "claude", turns)


def _telegram(data: dict) -> list[Thread]:
    chats = data.get("chats")
    listings = []
    if isinstance(chats, dict):
        listings = chats.get("list") or []
    elif isinstance(chats, list):
        listings = chats
    if not listings and isinstance(data.get("messages"), list):
        listings = [data]
    out = []
    for ch in listings:
        if not isinstance(ch, dict):
            continue
        title = str(ch.get("name") or ch.get("title") or "Telegram")[:80]
        msgs = _messages_of(ch.get("messages") or [])
        if msgs:
            out.append(Thread(title, "telegram", msgs))
    return out


def _from_jsonl(text: str) -> list[Thread]:
    turns: list[Turn] = []
    threads: list[Thread] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (obj.get("mapping") or obj.get("chat_messages") or obj.get("messages")):
            threads.extend(_from_obj(obj))
            continue
        msgs = _messages_of([obj] if "role" in obj or "from" in obj or "content" in obj else obj.get("conversations") or [])
        turns.extend(msgs)
    if threads:
        return threads
    if turns:
        return [Thread("jsonl", "jsonl", turns)]
    return []


def _from_csv(text: str) -> list[Thread]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip().strip('"').lower() for h in lines[0].split(",")]
    turns = []
    for ln in lines[1:]:
        cols = [c.strip().strip('"') for c in ln.split(",", 2)]
        row = {header[i]: cols[i] if i < len(cols) else "" for i in range(len(header))}
        content = row.get("content") or row.get("text") or row.get("message") or (cols[-1] if cols else "")
        role = row.get("role") or row.get("from") or row.get("sender") or "user"
        if content:
            turns.append(Turn(_role(role, ""), role, content[:4000]))
    return [Thread("csv", "csv", turns)] if turns else []


def _from_whatsapp(text: str, filename: str) -> list[Thread]:
    text = text.replace("\u202f", " ").replace("\u2009", " ").replace("\ufeff", "")
    hits = 0
    turns: list[Turn] = []
    cur: Turn | None = None
    for ln in text.splitlines():
        m = WA_LINE.match(ln.strip())
        if m:
            hits += 1
            name, body = m.group(3).strip(), m.group(4).strip()
            if cur:
                turns.append(cur)
                cur = None
            if SKIP_WA.search(body) or SKIP_WA.search(name):
                continue
            cur = Turn("user", name[:40], body, None)
        elif cur and ln.strip() and not WA_LINE.match(ln.strip()[:40] + (ln[40:] if len(ln) > 40 else "")):
            if not SKIP_WA.search(ln):
                cur.content = (cur.content + "\n" + ln.rstrip())[:4000]
    if cur:
        turns.append(cur)
    if hits < 3 or not turns:
        return []
    title = filename.rsplit("/", 1)[-1]
    title = re.sub(r"(_chat|\.txt$)", "", title, flags=re.I).strip(" _-") or "WhatsApp"
    # map most-frequent speaker after "You" as still user; assistant names rare in WA
    return [Thread(title[:80], "whatsapp", turns)]


def _from_markdown(text: str, filename: str) -> list[Thread]:
    turns: list[Turn] = []
    pat = re.compile(r"^\s*(?:\*\*)?(user|assistant|human|agi|chatgpt|claude|you|me)(?:\*\*)?\s*[:\-]\s*(.*)$", re.I)
    cur: Turn | None = None
    for ln in text.splitlines():
        m = pat.match(ln)
        if m:
            if cur:
                turns.append(cur)
            role = _role(m.group(1), "")
            cur = Turn(role, m.group(1), m.group(2))
        elif cur:
            cur.content = (cur.content + "\n" + ln)[:4000]
    if cur:
        turns.append(cur)
    if len(turns) < 2:
        return []
    return [Thread(filename[:80], "markdown", turns)]


def _messages_of(items: Any) -> list[Turn]:
    if isinstance(items, dict):
        items = items.get("messages") or items.get("conversation") or items.get("conversations") or []
    if not isinstance(items, list):
        return []
    out: list[Turn] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "from" in it and "value" in it:  # sharegpt
            role = _role(str(it.get("from")), str(it.get("from")))
            content = str(it.get("value") or "")
        else:
            role_raw = (
                it.get("role")
                or (it.get("author") or {}).get("role")
                or it.get("sender")
                or it.get("from")
                or it.get("from_id")
                or "user"
            )
            if isinstance(role_raw, dict):
                role_raw = role_raw.get("role") or role_raw.get("name") or "user"
            name = str(it.get("name") or it.get("from") or role_raw)
            content = it.get("content") or it.get("text") or it.get("message") or ""
            if isinstance(content, dict):
                parts = content.get("parts") or content.get("text") or ""
                if isinstance(parts, list):
                    content = "\n".join(p if isinstance(p, str) else "" for p in parts)
                else:
                    content = str(parts)
            elif isinstance(content, list):
                bits = []
                for p in content:
                    if isinstance(p, str):
                        bits.append(p)
                    elif isinstance(p, dict) and p.get("text"):
                        bits.append(str(p["text"]))
                content = "\n".join(bits)
            content = str(content)
            role = _role(str(role_raw), str(name))
        content = content.strip()
        if not content or content in {"", "<Media omitted>"}:
            continue
        out.append(Turn(role, str(name)[:40], content[:4000], _ts(it)))
        if len(out) >= MAX_MSGS:
            break
    return out


def _role(role: str, name: str) -> str:
    r = (role or "").lower().strip()
    n = (name or "").lower().strip()
    if r in ASSISTANT_ROLES or n in ASSISTANT_ROLES:
        return "agi"
    if r in USER_ROLES or n in USER_ROLES:
        return "user"
    return "user"


def _ts(it: dict) -> float | None:
    for k in ("create_time", "date", "date_unixtime", "timestamp", "ts"):
        v = it.get(k)
        if isinstance(v, (int, float)) and v > 1e8:
            return float(v)
        if isinstance(v, str) and v.isdigit():
            return float(v)
    return None


def _train_snips(threads: list[Thread]) -> list[str]:
    snips = []
    for th in threads:
        buf = []
        for t in th.turns:
            who = "user" if t.role != "agi" else "agi"
            buf.append(f"{who}: {t.content[:400]}")
            if len(buf) >= 6:
                snips.append("\n".join(buf)[:1800])
                buf = buf[-2:]
        if len(buf) >= 2:
            snips.append("\n".join(buf)[:1800])
        if len(snips) >= MAX_TRAIN:
            break
    return snips
