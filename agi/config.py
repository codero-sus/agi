from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep BLAS from oversubscribing the 2-core box.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parent.parent
PYTHON_ENV_PATH = ROOT / "python.env"
MODEL_DIR = Path(os.environ.get("AGI_MODEL_DIR", ROOT / "model"))
DATA_DIR = Path(os.environ.get("AGI_DATA_DIR", ROOT / "data"))
SKILLS_DIR = Path(os.environ.get("AGI_SKILLS_DIR", ROOT / "skills"))
VAULT_DIR = Path(os.environ.get("AGI_VAULT_DIR", DATA_DIR / "vault"))
DOCS_DIR = Path(os.environ.get("AGI_DOCS_DIR", DATA_DIR / "docs"))
WEB_DIR = Path(__file__).resolve().parent / "web"

GGUF_NAME = "model.gguf"
SAFETENSORS_NAME = "model.safetensors"
GGUF_PATH = MODEL_DIR / GGUF_NAME
SAFETENSORS_PATH = MODEL_DIR / SAFETENSORS_NAME
CONFIG_PATH = MODEL_DIR / "config.json"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

HOST = os.environ.get("AGI_HOST", "0.0.0.0")
PORT = int(os.environ.get("AGI_PORT", "8000"))

# CortexGPT defaults — small enough for CPU / 4GB RAM, trains in the background.
CORTEX_DIM = 256
CORTEX_LAYERS = 6
CORTEX_HEADS = 8
CORTEX_CTX = 256
CORTEX_VOCAB = 259  # 256 utf-8 bytes + BOS/EOS/PAD
CORTEX_LR = 3e-4
CORTEX_TRAIN_STEPS = 8  # blocking cycle steps; more happen in the background
CORTEX_HOT_STEPS = 0  # request path never blocks on SGD
CHECKPOINT_EVERY = 12
DREAM_IDLE_SEC = 20.0

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.95

# Refuse to mmap enormous dense weights into 4GB RAM.
MAX_NUMPY_PARAMS = 120_000_000
MAX_NUMPY_BYTES = 1_600_000_000

MEMORY_INDEX_CAP = 2000

# Optional speech backends (OpenAI-compatible). Cortex AGI stays the mind.
# AGI_LLM=local|ollama|hoster|openrouter
# AGI_OLLAMA_URL (default http://127.0.0.1:11434), AGI_OLLAMA_MODEL
# AGI_HOSTER_URL (default http://127.0.0.1:8624), AGI_HOSTER_MODEL, AGI_HOSTER_KEY / CORTEX_API_KEY
# OPENROUTER_API_KEY / AGI_OPENROUTER_KEY, AGI_OPENROUTER_MODEL (OpenRouter is https://openrouter.ai only)
# AGI_UPDATE_REPO (default codero-sus/agi), AGI_UPDATE_RESTART=0 to skip process restart after apply


def _resolve_python_path(raw: str) -> str:
    line = (raw or "").strip().strip('"').strip("'")
    if not line:
        return ""
    p = Path(line)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def python_from_env_file(path: Path | None = None) -> str:
    """Optional python.env at the project root. Empty string if missing."""
    path = path or PYTHON_ENV_PATH
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("python="):
            line = line.split("=", 1)[1]
        found = _resolve_python_path(line)
        if found:
            return found
    return ""


def python_executable() -> str:
    """2PY2 (portable Python) wins, then optional python.env, then this process."""
    portable = (os.environ.get("2PY2") or "").strip()
    if portable:
        return _resolve_python_path(portable) or portable
    from_file = python_from_env_file()
    if from_file:
        return from_file
    return sys.executable


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "events").mkdir(parents=True, exist_ok=True)
