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

## What “self-improvement” means here

After every turn CORTEX:

1. Stores **episodic memory** (the conversation)
2. Extracts **semantic facts** (`user name Ada`)
3. **Critiques** its own reply and may add a lesson / constitution clause
4. Takes **gradient steps** on CortexGPT and writes `model/model.safetensors`
5. Every few turns: **self-eval battery**, skill synthesis, extra training

Skills live in `skills/` as real Python files the agent can write.

Persistent mind state is under `data/` (SQLite + identity + goals).

## API

- `GET /` — control-plane UI
- `WS /ws` — streaming thoughts + tokens
- `POST /api/chat` — `{ "message": "..." }`
- `GET /api/state` — identity, memory, loss, skills
- `POST /api/improve` — force an improvement cycle
- `POST /v1/chat/completions` — OpenAI-compatible
- `GET /v1/models`

## Architecture

```
user
  → custom server (FastAPI)
    → cognition (understand → recall → plan → act → reflect)
      → memory / knowledge / tools / skills
      → ModelEngine (GGUF | safetensors | CortexGPT)
    → self-improvement loop
      → facts, lessons, constitution, skills, SGD, checkpoint
```
