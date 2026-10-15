# Model directory

The custom server looks here, in this order, for **speech**:

1. `model.gguf` — llama.cpp via `llama-cpp-python` when installed. GGUF metadata is always parsed.
2. `model.safetensors` — Llama / GPT-2 Hugging Face weights (forward-only numpy).
3. CortexGPT — the mind's own numpy transformer.

CortexGPT **always** trains in the background and checkpoints itself to:

- `cortex.safetensors` — weights
- `cortex.config.json` — architecture, step count, recent loss

Legacy `model.safetensors` files that are CortexGPT layouts are loaded once and rewritten to `cortex.safetensors`.

Optional companions for Hugging Face dumps:

- `config.json`
- `tokenizer.json`

Drop a GGUF or HF safetensors file here and restart (or `POST /api/reload-model`). Pick Ollama or Cortex LLMHoster in the hosts tab to chat through that model instead.
