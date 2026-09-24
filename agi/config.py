from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("AGI_MODEL_DIR", ROOT / "model"))
DATA_DIR = Path(os.environ.get("AGI_DATA_DIR", ROOT / "data"))
SKILLS_DIR = Path(os.environ.get("AGI_SKILLS_DIR", ROOT / "skills"))
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
CORTEX_TRAIN_STEPS = 24

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_K = 40

# Refuse to mmap enormous dense weights into 4GB RAM.
MAX_NUMPY_PARAMS = 120_000_000
MAX_NUMPY_BYTES = 1_600_000_000


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "events").mkdir(parents=True, exist_ok=True)
