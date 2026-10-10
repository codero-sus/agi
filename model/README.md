# Model directory

The custom server looks here, in this order:

1. `model.gguf` — llama.cpp via `llama-cpp-python` when installed. GGUF metadata is always parsed.
2. `model.safetensors` — Llama / GPT-2 / Qwen-style Hugging Face weights, **or** Cortex AGI's own numpy GPT.

Optional companions:

- `config.json` — Hugging Face architecture config
- `tokenizer.json` — Hugging Face tokenizer
- `model.config.json` — written by CortexGPT when it checkpoints itself

If neither weight file exists, Cortex AGI boots its own neural core and **writes** `model.safetensors` as it trains on your conversations.

Drop a GGUF or safetensors file here and restart (or `POST /api/reload-model`).
