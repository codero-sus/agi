# CORTEX

A local **AGI** with a custom inference server, a cognitive loop, and self-improvement that actually writes to disk.

CORTEX is not a wrapper around OpenAI. It loads weights from this repo:

| file | backend |
|---|---|
| `model/model.gguf` | llama.cpp (`llama-cpp-python`) |
| `model/model.safetensors` | numpy Llama / GPT-2, or CORTEX's own GPT |

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

## Features

- Teach: `learn this: Title — body` (persists under `data/knowledge.json`)
- Search memory (UI box or `GET /api/search?q=`)
- Forget facts: `forget that …`
- Convert units, time (UTC + IST), SHA-256, sandboxed Python
- Summarize the conversation, export transcript
- Dream replay: idle trainer re-learns old episodes
- Stop generation, restore history on reload

## What “self-improvement” means here

After every turn CORTEX:

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

## Architecture

```
user
  → custom server (FastAPI, gzip, worker-thread think)
    → cognition (understand → recall → plan → act → reflect)
      → numpy memory index / knowledge / tools / skills
      → ModelEngine (GGUF | safetensors | CortexGPT + KV cache)
    → background trainer (Adam, dream, debounce checkpoint)
```
