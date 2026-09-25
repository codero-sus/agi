"""Model discovery and a single ModelEngine facade.

Priority:
  1. model/model.gguf          (llama.cpp if installed; metadata always)
  2. model/model.safetensors   (external HF llama/gpt2, or CORTEX's own GPT)
  3. bootstrap CortexGPT       (trains online, writes model.safetensors)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from agi.config import (
    GGUF_PATH,
    MAX_NEW_TOKENS,
    MODEL_DIR,
    SAFETENSORS_PATH,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    ensure_dirs,
)
from agi.improve.trainer import BackgroundTrainer

from .cortex_gpt import CortexGPT
from .gguf_engine import LlamaCppEngine, parse_gguf
from .safetensors_engine import try_load_external_safetensors


@dataclass
class EngineInfo:
    source: str  # gguf | safetensors | cortex | bootstrap
    backend: str
    ready: bool
    detail: dict = field(default_factory=dict)
    warning: str | None = None

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "backend": self.backend,
            "ready": self.ready,
            "warning": self.warning,
            **self.detail,
        }


class ModelEngine:
    def __init__(self):
        ensure_dirs()
        self.cortex = CortexGPT()
        self.external = None
        self.info = EngineInfo("bootstrap", "cortex-gpt", True, {"params": self.cortex.param_count()})
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.trainer = BackgroundTrainer(self)
        self._discover()

    def _discover(self) -> None:
        if GGUF_PATH.exists():
            self._load_gguf(GGUF_PATH)
            return
        if SAFETENSORS_PATH.exists():
            self._load_safetensors(SAFETENSORS_PATH)
            return
        self.info = EngineInfo(
            source="bootstrap",
            backend="cortex-gpt",
            ready=True,
            detail={
                "params": self.cortex.param_count(),
                "steps": self.cortex.steps,
                "note": "No model.gguf or model.safetensors found. "
                "CORTEX is running its own neural core and will write "
                "model/model.safetensors as it trains.",
            },
        )

    def reload(self) -> EngineInfo:
        self.external = None
        self._discover()
        return self.info

    def _load_gguf(self, path: Path) -> None:
        try:
            meta = parse_gguf(path)
            summary = meta.summary()
        except Exception as e:
            self.info = EngineInfo("gguf", "unreadable", False, warning=str(e))
            return
        try:
            self.external = LlamaCppEngine(path, meta)
            self.info = EngineInfo("gguf", "llama.cpp", True, summary)
        except ImportError:
            self.info = EngineInfo(
                "gguf",
                "metadata-only",
                False,
                summary,
                warning="Found model/model.gguf but llama-cpp-python is not installed. "
                "Install it to run GGUF inference, or drop a safetensors checkpoint instead.",
            )
        except Exception as e:
            self.info = EngineInfo("gguf", "llama.cpp", False, summary, warning=str(e))

    def _load_safetensors(self, path: Path) -> None:
        kind, runner, info = try_load_external_safetensors(path)
        if kind == "cortex-gpt":
            loaded = CortexGPT.load(path)
            if loaded is not None:
                self.cortex = loaded
                self.info = EngineInfo(
                    "safetensors",
                    "cortex-gpt",
                    True,
                    {
                        **info,
                        "params": self.cortex.param_count(),
                        "steps": self.cortex.steps,
                        "architecture": "cortex-gpt",
                    },
                )
                return
        if runner is not None and kind in ("llama", "gpt2"):
            self.external = runner
            self.info = EngineInfo("safetensors", f"numpy-{kind}", True, info)
            return
        loaded = CortexGPT.load(path)
        if loaded is not None:
            self.cortex = loaded
            self.info = EngineInfo("safetensors", "cortex-gpt", True, {**info, "params": self.cortex.param_count()})
            return
        self.info = EngineInfo(
            "safetensors",
            "unusable",
            False,
            info,
            warning=info.get("error") or "Could not load model.safetensors",
        )
        self.info.ready = True
        self.info.backend = "cortex-gpt (fallback)"

    def has_external_lm(self) -> bool:
        return self.external is not None

    def generate(self, prompt: str, max_new: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE) -> str:
        self.stop.clear()
        with self.lock:
            if self.external is not None:
                try:
                    return self.external.generate(prompt, max_new=max_new, temperature=temperature, top_k=TOP_K)
                except TypeError:
                    return self.external.generate(prompt, max_new=max_new, temperature=temperature)
            return self.cortex.generate(
                prompt,
                max_new=min(max_new, 96),
                temperature=temperature,
                top_k=TOP_K,
                top_p=TOP_P,
                stop=self.stop,
            )

    def stream(self, prompt: str, max_new: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE) -> Iterator[str]:
        ext = self.external
        if ext is not None and hasattr(ext, "stream"):
            yield from ext.stream(prompt, max_new=max_new, temperature=temperature)
            return
        text = self.generate(prompt, max_new=max_new, temperature=temperature)
        buf = ""
        for ch in text:
            buf += ch
            if ch in " \n.,;:!?":
                yield buf
                buf = ""
        if buf:
            yield buf

    def train_on(self, text: str, steps: int = 4, blocking: bool = False) -> float:
        if not text or len(text) < 8:
            return 0.0
        if not blocking:
            self.trainer.submit(text, steps)
            return float(self.trainer.last_loss or 0.0)
        loss = 0.0
        n = 0
        for _ in range(max(1, steps)):
            loss += self.cortex.train_step(text)
            n += 1
        avg = loss / max(n, 1)
        self.trainer._dirty += n
        if self.trainer._dirty >= 8:
            self.trainer.flush_save()
        return avg

    def request_stop(self) -> None:
        self.stop.set()

    def neural_stats(self) -> dict:
        hist = self.cortex.loss_history
        return {
            "steps": self.cortex.steps,
            "params": self.cortex.param_count(),
            "last_loss": hist[-1] if hist else None,
            "loss_history": hist[-80:],
            "trainer": self.trainer.snapshot(),
        }


def load_engine() -> ModelEngine:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return ModelEngine()
