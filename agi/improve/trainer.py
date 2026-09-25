"""Background SGD / dream loop so the request path never waits on training."""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Callable

from agi.config import CHECKPOINT_EVERY, CORTEX_TRAIN_STEPS, DREAM_IDLE_SEC, SAFETENSORS_PATH

if TYPE_CHECKING:
    from agi.inference.loader import ModelEngine


class BackgroundTrainer:
    def __init__(self, engine: "ModelEngine"):
        self.engine = engine
        self.q: queue.Queue[tuple[str, int]] = queue.Queue(maxsize=128)
        self.dream_source: Callable[[], str] | None = None
        self.busy = False
        self.queued = 0
        self.done_steps = 0
        self.last_loss = 0.0
        self.last_save = 0.0
        self._dirty = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cortex-trainer")
        self._thread.start()

    def submit(self, text: str, steps: int = 4) -> bool:
        if not text or len(text) < 8:
            return False
        try:
            self.q.put_nowait((text, max(1, steps)))
            self.queued += 1
            return True
        except queue.Full:
            return False

    def snapshot(self) -> dict:
        return {
            "busy": self.busy,
            "queue": self.q.qsize(),
            "done_steps": self.done_steps,
            "last_loss": self.last_loss,
        }

    def flush_save(self) -> None:
        try:
            self.engine.cortex.save(SAFETENSORS_PATH)
            self.last_save = time.time()
            self._dirty = 0
        except Exception:
            pass

    def _run(self) -> None:
        last_dream = time.time()
        while not self._stop.is_set():
            try:
                text, steps = self.q.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                if now - last_dream >= DREAM_IDLE_SEC:
                    last_dream = now
                    self._dream()
                if self._dirty and now - self.last_save > 15:
                    self.flush_save()
                continue
            self.busy = True
            try:
                loss_acc = 0.0
                n = 0
                for _ in range(steps):
                    loss_acc += self.engine.cortex.train_step(text)
                    n += 1
                    self.done_steps += 1
                    self._dirty += 1
                if n:
                    self.last_loss = loss_acc / n
                if self._dirty >= CHECKPOINT_EVERY:
                    self.flush_save()
            except Exception:
                pass
            finally:
                self.busy = False

    def _dream(self) -> None:
        src = self.dream_source
        if src is None:
            return
        try:
            text = src()
        except Exception:
            return
        if not text:
            return
        self.busy = True
        try:
            for _ in range(max(2, CORTEX_TRAIN_STEPS // 4)):
                self.last_loss = self.engine.cortex.train_step(text)
                self.done_steps += 1
                self._dirty += 1
        except Exception:
            pass
        finally:
            self.busy = False
