# Cortex AGI

A local **AGI** with a custom inference server, a cognitive loop, and self-improvement that actually writes to disk.

Cortex AGI is not a wrapper around OpenAI. It loads weights from this repo:

| file | backend |
|---|---|
| `model/model.gguf` | llama.cpp (`llama-cpp-python`) |
| `model/model.safetensors` | numpy Llama / GPT-2, or Cortex AGI's own GPT |

If neither file is present, it boots **CortexGPT** (a numpy transformer), trains on every conversation, and checkpoints itself to `model/model.safetensors`.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m agi
```

Then open the control plane at `http://localhost:8000`.

Optional GGUF runtime:

```bash
pip install llama-cpp-python
```

Drop `model.gguf` or `model.safetensors` into `model/` and restart, or `POST /api/reload-model`.

## Efficiency (v0.2)

- **KV-cached generation** — the prompt is encoded once
- **Adam** instead of vanilla SGD, reused grad buffers
- **Background trainer** — replies never wait on weight updates
- **Debounced checkpoints** — safetensors writes every N steps, not every token
- **Numpy memory index** — recall is a matrix-vector product
- **Think runs in a worker thread** so the WebSocket event loop stays live
- gzip + static cache headers

## Chain of thought (v0.3)

Hard questions run **System 2** before speaking:

1. **Parse** — restate the question and what would count as an answer
2. **Strategy** — causal, mechanism, compare, counterfactual, first-principles, plan, retrieve
3. **Decompose** — 2–4 subquestions
4. **Retrieve / deduce** — knowledge, memory, tools
5. **Hypothesize** — competing candidates, scored
6. **Critique** — devil's advocate
7. **Decide** — commit with a confidence

The chain streams live in the control plane and is attached to every reply. Fast facts (math, time, convert) stay System 1 with a short trace.

## Import your other lives (v0.8)

Drop an export onto the desk. Cortex AGI parses it, remembers the turns, opens a thread, and **trains the neural core** on the dialogue.

| export | how |
|---|---|
| WhatsApp | Settings → Chat → Export chat → `.txt` or `.zip` |
| ChatGPT | Settings → Data controls → Export → `conversations.json` |
| Claude | conversation JSON download |
| Telegram | Export chat history → `result.json` |
| generic | OpenAI `messages` JSON, ShareGPT, JSONL, CSV |

`POST /api/import` or the vault tab **import chats**. Caps: 25MB, 5000 turns. Private URLs stay blocked; this is local-only.

## Steal your own context (v0.9)

The **import** tab has a prompt you paste into any other model. It replies with `cortex_export` JSON. Paste that JSON back. Cortex AGI absorbs it.

```
GET  /api/import/prompt
POST /api/import/json   { "text": "{ ... cortex_export ... }" }
```

## Mouths (v0.10)

Cortex AGI stays the mind. Optional **OpenAI-compatible** backends only speak — they do not train, they do not own memory.

| mouth | default | auth |
|---|---|---|
| local CortexGPT / gguf / safetensors | `model/` | none |
| [Ollama](https://ollama.com) | `http://127.0.0.1:11434` | none |
| [Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster) | `http://127.0.0.1:8624` | `CORTEX_API_KEY` if set |
| [OpenRouter](https://openrouter.ai) | `https://openrouter.ai/api/v1` only | `OPENROUTER_API_KEY` |

Pick one in the **hosts** tab. Keys are stored in `data/hosts.json`, masked on `GET`, never logged.

```
GET  /api/hosts
POST /api/hosts          { "active": "ollama", "ollama": { "url", "model" } }
POST /api/hosts/probe
```

Env: `AGI_LLM=ollama|hoster|openrouter|local`, `AGI_OLLAMA_URL`, `AGI_HOSTER_URL`, `OPENROUTER_API_KEY`. Ollama and the hoster may be localhost or LAN; OpenRouter is pinned to `openrouter.ai` over HTTPS. Local weights still train.

## Updater (v0.11)

The **mind** tab (and settings) can fast-forward this checkout from GitHub. Memory under `data/` and trained weights stay. Origin must be `github.com/codero-sus/agi`. Fast-forward only — dirty or diverged trees are refused. After apply the process restarts (`AGI_UPDATE_RESTART=0` to skip).

```
GET  /api/update
POST /api/update/check
POST /api/update          { "restart": true }
```

Say `check for updates` or `update yourself`. `/update` in the composer.

From the repo root:

```bash
./updater.sh          # macOS / Linux
updater.bat           # Windows
```

Python is read from **`python.env`** (`PYTHON=...`). Point it at your venv. Same rules: GitHub `codero-sus/agi` only, fast-forward, `data/` and weights stay. Restart `"$PYTHON" -m agi` after it finishes.

## Agents you can create (v0.7)

Agentic AI here is not a chatbot with plugins. You **spawn named agents** that are slices of this mind:

```
create agent Scout — mission: watch the vault tools: vault, note, knowledge
@Scout what did we ingest
run Researcher: photosynthesis
```

- Each agent has a **mission** and a **tool whitelist** (knowledge, vault, memory, math, python, wiki, fetch, note, task, research, hash, now)
- Seed crew: Researcher, Critic, Tutor, Operator
- Runs write a document and train the neural core
- No bash, no MCP, no silent root — constitution travels with every spawn
- API: `GET/POST /api/agents`, `POST /api/agents/{name}/run`

Pick an agent in the composer or the crew tab. Right-click to retire one.

## Agent & compare (v0.6)

Odysseus's headline is **agents + compare + cookbook**. Those are wrappers. Cortex AGI owns them:

- **`do:`** — a bounded tool loop (knowledge, vault, memory, math, public URLs). No bash, no MCP, no silent root. The trace is a document that trains the core.
- **`compare A and B`** — one chain, two characterizations, a split. Not five vendor APIs side-by-side.
- **Core** — one growing `model.gguf` / `model.safetensors`, not a catalogue of 270 downloads.
- **Desk rail** — threads, documents, and open tasks in one column. Odysseus splits them into apps.

## Desk (v0.5) — built to beat Odysseus

[Odysseus](https://github.com/odysseus-dev/odysseus) is a self-hosted **suite**: chat wrapping Ollama, a research agent wrapping the web, a document app, email, calendar, 270-model cookbook. Useful. Not an AGI.

Cortex AGI is one mind that does the jobs people actually open Odysseus for:

| Odysseus | Cortex AGI |
|---|---|
| Wraps Ollama / OpenAI / vLLM | **Is** the model (`model.gguf` / `model.safetensors` / CortexGPT) |
| Deep research as a separate agent | System-2 research → cited report → **taught back into weights** |
| Documents as an editor app | Documents the mind owns; they train it |
| Tasks / calendar as productivity | Tasks as working memory, spawned from research |
| Skills via MCP servers | Skills as Python the AGI writes |
| Compare five APIs | Compare hypotheses in one chain |
| Cookbook of 270 downloads | One core that grows |

Say `research this: photosynthesis`, `note: …`, `todo: …`. Reports land in **docs**, next actions in **todo**, evidence in the live chain.

## Workspace (v0.4) — built to beat Open WebUI

Open WebUI is a wrapper around someone else's model. Cortex AGI is the model, the mind, and the UI:

- **Threads** with search, pin-worthy titles, JSON export
- **File vault** — drop `.md/.py/.json` onto the page; they become knowledge
- **Artifacts** — code blocks hop into a side panel with copy + HTML preview
- **Voice** — mic in, spoken replies out (browser APIs)
- **Slash commands** and a **⌘K command palette**
- **Focus mode** (Ctrl+.) — hide chrome, keep the mind
- Shared long-term memory across threads (WebUI chats are amnesiac silos)

## Features

- Teach: `learn this: Title — body` (persists under `data/knowledge.json`)
- Search memory (UI box or `GET /api/search?q=`)
- Forget facts: `forget that …`
- Convert units, time (UTC + IST), SHA-256, sandboxed Python
- Summarize the conversation, export transcript
- Dream replay: idle trainer re-learns old episodes
- Stop generation, restore history on reload

## What “self-improvement” means here

After every turn Cortex AGI:

1. Stores **episodic memory** (the conversation)
2. Extracts **semantic facts** (`user name Ada`)
3. **Critiques** its own reply and may add a lesson / constitution clause
4. Queues **Adam steps** on CortexGPT (background) → `model/model.safetensors`
5. Every few turns: **self-eval battery**, skill synthesis, extra training

Skills live in `skills/` as real Python files the agent can write.

Persistent mind state is under `data/` (SQLite + identity + goals + taught articles).

## API

- `GET /` — control-plane UI
- `WS /ws` — streaming thoughts + tokens (`ping`, `stop`, `improve`)
- `POST /api/chat` — `{ "message": "..." }`
- `GET /api/state` — identity, memory, loss, skills
- `GET /api/history` — recent dialogue
- `GET /api/search?q=` — memory + knowledge
- `POST /api/teach` — `{ "title", "body" }`
- `POST /api/forget?q=` — drop matching facts
- `GET /api/export` — transcript
- `POST /api/improve` — force an improvement cycle
- `POST /v1/chat/completions` — OpenAI-compatible
- `GET /v1/models`
- `GET/POST /api/hosts` — attach Ollama / LLMHoster / OpenRouter as a mouth
- `POST /api/hosts/probe`
- `GET /api/update` — current SHA vs origin
- `POST /api/update` — fast-forward source, then restart

## Architecture

```
user
  → custom server (FastAPI, gzip, worker-thread think)
    → cognition (understand → recall → plan → act → reflect)
      → numpy memory index / knowledge / tools / skills
      → ModelEngine (GGUF | safetensors | CortexGPT + optional Ollama/Hoster/OpenRouter mouth)
    → background trainer (Adam, dream, debounce checkpoint)
```
