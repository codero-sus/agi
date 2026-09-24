"""A compact GPT that lives in numpy, trains online, and serializes to safetensors.

This is the AGI's own neural core — not a wrapper around someone else's server.
When no external GGUF/safetensors checkpoint is present, CORTEX trains this
model on every conversation and writes `model/model.safetensors`.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from agi.config import (
    CORTEX_CTX,
    CORTEX_DIM,
    CORTEX_HEADS,
    CORTEX_LAYERS,
    CORTEX_LR,
    SAFETENSORS_PATH,
    TOKENIZER_PATH,
)

from .tokenizer import BYTE_VOCAB, ByteTokenizer


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    tanh = np.tanh(inner)
    d_inner = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
    d_tanh = (1.0 - tanh * tanh) * d_inner
    return 0.5 * (1.0 + tanh) + 0.5 * x * d_tanh


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x, -60, 60))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


@dataclass
class CortexConfig:
    vocab_size: int = BYTE_VOCAB
    n_embd: int = CORTEX_DIM
    n_layer: int = CORTEX_LAYERS
    n_head: int = CORTEX_HEADS
    n_ctx: int = CORTEX_CTX
    architecture: str = "cortex-gpt"

    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def _randn(*shape: int, scale: float = 0.02) -> np.ndarray:
    return (np.random.randn(*shape).astype(np.float32) * scale)


class CortexGPT:
    """Pre-LN GPT with causal attention. Forward, sample, and SGD all in numpy."""

    def __init__(self, config: CortexConfig | None = None, params: dict | None = None):
        self.config = config or CortexConfig()
        self.tokenizer = ByteTokenizer()
        self.lock = threading.Lock()
        self.steps = 0
        self.loss_history: list[float] = []
        self.params: dict[str, np.ndarray] = params or self._init_params()

    def _init_params(self) -> dict[str, np.ndarray]:
        c = self.config
        p: dict[str, np.ndarray] = {
            "tok_emb": _randn(c.vocab_size, c.n_embd),
            "pos_emb": _randn(c.n_ctx, c.n_embd, scale=0.01),
            "ln_f.w": np.ones((c.n_embd,), dtype=np.float32),
            "ln_f.b": np.zeros((c.n_embd,), dtype=np.float32),
        }
        for i in range(c.n_layer):
            p[f"h.{i}.ln1.w"] = np.ones((c.n_embd,), dtype=np.float32)
            p[f"h.{i}.ln1.b"] = np.zeros((c.n_embd,), dtype=np.float32)
            p[f"h.{i}.attn.wqkv"] = _randn(c.n_embd, 3 * c.n_embd)
            p[f"h.{i}.attn.wo"] = _randn(c.n_embd, c.n_embd)
            p[f"h.{i}.ln2.w"] = np.ones((c.n_embd,), dtype=np.float32)
            p[f"h.{i}.ln2.b"] = np.zeros((c.n_embd,), dtype=np.float32)
            p[f"h.{i}.mlp.w1"] = _randn(c.n_embd, 4 * c.n_embd)
            p[f"h.{i}.mlp.b1"] = np.zeros((4 * c.n_embd,), dtype=np.float32)
            p[f"h.{i}.mlp.w2"] = _randn(4 * c.n_embd, c.n_embd, scale=0.02 / math.sqrt(2 * c.n_layer))
            p[f"h.{i}.mlp.b2"] = np.zeros((c.n_embd,), dtype=np.float32)
        return p

    # ------------------------------------------------------------------ forward
    def _ln(self, x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        xh = (x - mean) * inv
        return w * xh + b, {"xh": xh, "inv": inv, "w": w}

    def _ln_backward(self, dy: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xh, inv, w = cache["xh"], cache["inv"], cache["w"]
        dw = (dy * xh).sum(axis=tuple(range(dy.ndim - 1)))
        db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        dxh = dy * w
        n = xh.shape[-1]
        dx = inv * (dxh - dxh.mean(axis=-1, keepdims=True) - xh * (dxh * xh).mean(axis=-1, keepdims=True) * n / n)
        return dx, dw.astype(np.float32), db.astype(np.float32)

    def _block_forward(self, x: np.ndarray, i: int, mask: np.ndarray):
        p = self.params
        c = self.config
        T, D = x.shape[-2], x.shape[-1]
        H = c.n_head
        dh = D // H

        ln1, ln1c = self._ln(x, p[f"h.{i}.ln1.w"], p[f"h.{i}.ln1.b"])
        qkv = ln1 @ p[f"h.{i}.attn.wqkv"]
        qkv = qkv.reshape(T, 3, H, dh)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [T, H, dh]
        q = np.transpose(q, (1, 0, 2))
        k = np.transpose(k, (1, 0, 2))
        v = np.transpose(v, (1, 0, 2))
        att = (q @ np.transpose(k, (0, 2, 1))) / math.sqrt(dh)
        att = att + mask
        prob = _softmax(att, axis=-1)
        y = prob @ v  # [H, T, dh]
        y = np.transpose(y, (1, 0, 2)).reshape(T, D)
        attn_out = y @ p[f"h.{i}.attn.wo"]
        x2 = x + attn_out

        ln2, ln2c = self._ln(x2, p[f"h.{i}.ln2.w"], p[f"h.{i}.ln2.b"])
        h = ln2 @ p[f"h.{i}.mlp.w1"] + p[f"h.{i}.mlp.b1"]
        h_act = _gelu(h)
        mlp = h_act @ p[f"h.{i}.mlp.w2"] + p[f"h.{i}.mlp.b2"]
        out = x2 + mlp
        cache = {
            "x": x, "ln1": ln1, "ln1c": ln1c, "qkv": qkv, "q": q, "k": k, "v": v,
            "att": att, "prob": prob, "y": y, "attn_out": attn_out, "x2": x2,
            "ln2": ln2, "ln2c": ln2c, "h": h, "h_act": h_act, "mlp": mlp,
        }
        return out, cache

    def forward(self, idx: np.ndarray) -> tuple[np.ndarray, dict]:
        """idx: [T] int32 -> logits [T, vocab]."""
        c = self.config
        p = self.params
        T = idx.shape[0]
        tok = p["tok_emb"][idx]
        pos = p["pos_emb"][:T]
        x = tok + pos
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
        caches = []
        for i in range(c.n_layer):
            x, cache = self._block_forward(x, i, mask)
            caches.append(cache)
        xn, lnc = self._ln(x, p["ln_f.w"], p["ln_f.b"])
        logits = xn @ p["tok_emb"].T  # weight tying
        return logits, {"idx": idx, "x": x, "xn": xn, "lnc": lnc, "blocks": caches, "tok": tok}

    def generate(self, prompt: str, max_new: int = 80, temperature: float = 0.8, top_k: int = 40) -> str:
        ids = self.tokenizer.encode(prompt)
        ids = ids[-self.config.n_ctx :]
        if not ids:
            ids = [self.tokenizer.bos_id]
        out: list[int] = []
        with self.lock:
            for _ in range(max_new):
                ctx = np.asarray(ids[-self.config.n_ctx :], dtype=np.int32)
                logits, _ = self.forward(ctx)
                logits = logits[-1] / max(temperature, 1e-6)
                if top_k and top_k < logits.shape[0]:
                    thresh = np.partition(logits, -top_k)[-top_k]
                    logits = np.where(logits < thresh, -1e9, logits)
                probs = _softmax(logits)
                nxt = int(np.random.choice(len(probs), p=probs.astype(np.float64)))
                if nxt == self.tokenizer.eos_id:
                    break
                ids.append(nxt)
                out.append(nxt)
                if len(out) > 8 and len(set(out[-8:])) == 1:
                    break
        return self.tokenizer.decode(out)

    # ------------------------------------------------------------------ train
    def train_step(self, text: str, lr: float = CORTEX_LR) -> float:
        ids = self.tokenizer.encode(text)
        if len(ids) < 4:
            return 0.0
        ctx = self.config.n_ctx
        if len(ids) > ctx + 1:
            start = np.random.randint(0, len(ids) - ctx - 1)
            ids = ids[start : start + ctx + 1]
        x = np.asarray(ids[:-1], dtype=np.int32)
        y = np.asarray(ids[1:], dtype=np.int32)
        with self.lock:
            loss = self._sgd(x, y, lr)
        self.steps += 1
        self.loss_history.append(float(loss))
        if len(self.loss_history) > 500:
            self.loss_history = self.loss_history[-500:]
        return float(loss)

    def _sgd(self, idx: np.ndarray, targets: np.ndarray, lr: float) -> float:
        logits, cache = self.forward(idx)
        T, V = logits.shape
        probs = _softmax(logits, axis=-1)
        nll = -np.log(probs[np.arange(T), targets] + 1e-9)
        loss = float(nll.mean())

        dlogits = probs
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T

        p = self.params
        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in p.items()}

        # tied lm head: logits = xn @ tok_emb.T
        xn = cache["xn"]
        grads["tok_emb"] += dlogits.T @ xn
        dxn = dlogits @ p["tok_emb"]
        dx, dw, db = self._ln_backward(dxn, cache["lnc"])
        grads["ln_f.w"] += dw
        grads["ln_f.b"] += db

        c = self.config
        H = c.n_head
        D = c.n_embd
        dh = D // H

        for i in reversed(range(c.n_layer)):
            b = cache["blocks"][i]
            # residual mlp
            dmlp = dx
            dx2 = dx
            grads[f"h.{i}.mlp.b2"] += dmlp.sum(axis=0)
            grads[f"h.{i}.mlp.w2"] += b["h_act"].T @ dmlp
            dh_act = dmlp @ p[f"h.{i}.mlp.w2"].T
            dh_pre = dh_act * _gelu_grad(b["h"])
            grads[f"h.{i}.mlp.b1"] += dh_pre.sum(axis=0)
            grads[f"h.{i}.mlp.w1"] += b["ln2"].T @ dh_pre
            dln2 = dh_pre @ p[f"h.{i}.mlp.w1"].T
            dln2x, dw, db = self._ln_backward(dln2, b["ln2c"])
            grads[f"h.{i}.ln2.w"] += dw
            grads[f"h.{i}.ln2.b"] += db
            dx2 = dx2 + dln2x

            d_attn_out = dx2
            dx = dx2
            grads[f"h.{i}.attn.wo"] += b["y"].T @ d_attn_out
            dy = d_attn_out @ p[f"h.{i}.attn.wo"].T
            dy_h = dy.reshape(T, H, dh).transpose(1, 0, 2)  # [H,T,dh]
            dprob = dy_h @ np.transpose(b["v"], (0, 2, 1))  # [H,T,T]
            dv = np.transpose(b["prob"], (0, 2, 1)) @ dy_h
            # softmax backward
            sp = b["prob"]
            datt = sp * (dprob - (dprob * sp).sum(axis=-1, keepdims=True))
            datt /= math.sqrt(dh)
            dq = datt @ b["k"]
            dk = np.transpose(datt, (0, 2, 1)) @ b["q"]
            dq_t = dq.transpose(1, 0, 2)  # [T,H,dh]
            dk_t = dk.transpose(1, 0, 2)
            dv_t = dv.transpose(1, 0, 2)
            dqkv = np.stack([dq_t, dk_t, dv_t], axis=1).reshape(T, 3 * D)
            grads[f"h.{i}.attn.wqkv"] += b["ln1"].T @ dqkv
            dln1 = dqkv @ p[f"h.{i}.attn.wqkv"].T
            dln1x, dw, db = self._ln_backward(dln1, b["ln1c"])
            grads[f"h.{i}.ln1.w"] += dw
            grads[f"h.{i}.ln1.b"] += db
            dx = dx + dln1x

        # embeddings
        dtok = dx
        dpos = dx
        np.add.at(grads["tok_emb"], idx, dtok)
        grads["pos_emb"][:T] += dpos

        # clip + step
        norm = math.sqrt(sum(float(np.square(g).sum()) for g in grads.values()) + 1e-12)
        scale = 1.0 if norm < 1.0 else 1.0 / norm
        for k, g in grads.items():
            p[k] -= (lr * scale * g).astype(np.float32)
        return loss

    # ------------------------------------------------------------------ io
    def save(self, path: Path | None = None) -> Path:
        path = path or SAFETENSORS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.astype(np.float32) for k, v in self.params.items()}
        try:
            from safetensors.numpy import save_file

            save_file(tensors, str(path))
        except Exception:
            np.savez(path.with_suffix(".npz"), **tensors)
        meta = asdict(self.config)
        meta["steps"] = self.steps
        meta["loss_history"] = self.loss_history[-100:]
        path.with_suffix(".config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "CortexGPT | None":
        cfg_path = path.with_suffix(".config.json")
        alt_cfg = path.parent / "cortex.config.json"
        config = CortexConfig()
        steps = 0
        hist: list[float] = []
        for p in (cfg_path, alt_cfg, path.parent / "config.json"):
            if p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if raw.get("architecture") == "cortex-gpt" or "n_embd" in raw:
                        config = CortexConfig(
                            vocab_size=int(raw.get("vocab_size", BYTE_VOCAB)),
                            n_embd=int(raw.get("n_embd", CORTEX_DIM)),
                            n_layer=int(raw.get("n_layer", CORTEX_LAYERS)),
                            n_head=int(raw.get("n_head", CORTEX_HEADS)),
                            n_ctx=int(raw.get("n_ctx", CORTEX_CTX)),
                        )
                        steps = int(raw.get("steps", 0))
                        hist = list(raw.get("loss_history") or [])
                        break
                except Exception:
                    continue
        tensors = _load_numpy_tensors(path)
        if tensors is None:
            return None
        if "tok_emb" not in tensors:
            return None
        model = cls(config=config, params=tensors)
        model.steps = steps
        model.loss_history = hist
        return model

    def param_count(self) -> int:
        return int(sum(v.size for v in self.params.values()))


def _load_numpy_tensors(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    if path.suffix == ".npz":
        data = np.load(path)
        return {k: data[k] for k in data.files}
    try:
        from safetensors.numpy import load_file

        return load_file(str(path))
    except Exception:
        try:
            from safetensors import safe_open

            out = {}
            with safe_open(str(path), framework="numpy") as f:
                for k in f.keys():
                    out[k] = f.get_tensor(k)
            return out
        except Exception:
            return None
