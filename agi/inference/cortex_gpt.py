"""A compact GPT that lives in numpy, trains online, and serializes to safetensors.

Efficiency:
  - KV-cached generation (prompt encoded once)
  - Adam instead of vanilla SGD (fewer steps to learn)
  - reused grad / moment buffers
  - cached causal masks
  - GELU without np.power
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
)

from .tokenizer import BYTE_VOCAB, ByteTokenizer

_GELU_C = 0.7978845608028654  # sqrt(2/pi)
_NEG_INF = np.float32(-1e9)


def _gelu(x: np.ndarray) -> np.ndarray:
    x3 = x * x * x
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = _GELU_C * (x + 0.044715 * (x * x * x))
    tanh = np.tanh(inner)
    d_inner = _GELU_C * (1.0 + 0.134145 * (x * x))
    d_tanh = (1.0 - tanh * tanh) * d_inner
    return 0.5 * (1.0 + tanh) + 0.5 * x * d_tanh


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x, -60, 60))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def _sample_token(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> int:
    logits = logits / max(float(temperature), 1e-6)
    if top_k and top_k < logits.shape[0]:
        thresh = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < thresh, _NEG_INF, logits)
    probs = _softmax(logits).astype(np.float64)
    if 0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cdf = np.cumsum(probs[order])
        cut = int(np.searchsorted(cdf, top_p, side="right"))
        keep = order[: max(1, cut + 1)]
        trimmed = np.zeros_like(probs)
        trimmed[keep] = probs[keep]
        probs = trimmed
    s = float(probs.sum())
    if not np.isfinite(s) or s <= 0:
        probs = np.full(logits.shape[0], 1.0 / logits.shape[0])
    else:
        probs /= s
    return int(np.random.choice(logits.shape[0], p=probs))


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
    return np.random.randn(*shape).astype(np.float32) * np.float32(scale)


class CortexGPT:
    """Pre-LN GPT with causal attention. Forward, sample, and Adam all in numpy."""

    def __init__(self, config: CortexConfig | None = None, params: dict | None = None):
        self.config = config or CortexConfig()
        self.tokenizer = ByteTokenizer()
        self.lock = threading.Lock()
        self.steps = 0
        self.loss_history: list[float] = []
        self.params: dict[str, np.ndarray] = params or self._init_params()
        self._grads: dict[str, np.ndarray] | None = None
        self._m: dict[str, np.ndarray] | None = None
        self._v: dict[str, np.ndarray] | None = None
        self._mask_cache: dict[int, np.ndarray] = {}

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

    def _zeros_like_params(self) -> dict[str, np.ndarray]:
        return {k: np.zeros_like(v) for k, v in self.params.items()}

    def _grad_buf(self) -> dict[str, np.ndarray]:
        if self._grads is None or self._grads.keys() != self.params.keys():
            self._grads = self._zeros_like_params()
        else:
            for g in self._grads.values():
                g.fill(0)
        return self._grads

    def _moments(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if self._m is None or self._m.keys() != self.params.keys():
            self._m = self._zeros_like_params()
            self._v = self._zeros_like_params()
        return self._m, self._v  # type: ignore[return-value]

    def _mask(self, t: int) -> np.ndarray:
        m = self._mask_cache.get(t)
        if m is None:
            m = np.triu(np.full((t, t), _NEG_INF, dtype=np.float32), k=1)
            self._mask_cache[t] = m
            if len(self._mask_cache) > 64:
                self._mask_cache.clear()
                self._mask_cache[t] = m
        return m

    # ------------------------------------------------------------------ layernorm
    def _ln(self, x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        xh = (x - mean) * inv
        return w * xh + b, {"xh": xh, "inv": inv, "w": w}

    def _ln_backward(self, dy: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xh, inv, w = cache["xh"], cache["inv"], cache["w"]
        reduce = tuple(range(dy.ndim - 1))
        dw = (dy * xh).sum(axis=reduce)
        db = dy.sum(axis=reduce)
        dxh = dy * w
        dx = inv * (
            dxh
            - dxh.mean(axis=-1, keepdims=True)
            - xh * (dxh * xh).mean(axis=-1, keepdims=True)
        )
        return dx, dw.astype(np.float32), db.astype(np.float32)

    # ------------------------------------------------------------------ train forward
    def _block_forward(self, x: np.ndarray, i: int, mask: np.ndarray):
        p = self.params
        c = self.config
        t, d = x.shape[-2], x.shape[-1]
        h = c.n_head
        dh = d // h

        ln1, ln1c = self._ln(x, p[f"h.{i}.ln1.w"], p[f"h.{i}.ln1.b"])
        qkv = ln1 @ p[f"h.{i}.attn.wqkv"]
        qkv_v = qkv.reshape(t, 3, h, dh)
        q = np.transpose(qkv_v[:, 0], (1, 0, 2))
        k = np.transpose(qkv_v[:, 1], (1, 0, 2))
        v = np.transpose(qkv_v[:, 2], (1, 0, 2))
        att = (q @ np.transpose(k, (0, 2, 1))) * np.float32(1.0 / math.sqrt(dh))
        att = att + mask
        prob = _softmax(att, axis=-1)
        y = prob @ v
        y = np.transpose(y, (1, 0, 2)).reshape(t, d)
        attn_out = y @ p[f"h.{i}.attn.wo"]
        x2 = x + attn_out

        ln2, ln2c = self._ln(x2, p[f"h.{i}.ln2.w"], p[f"h.{i}.ln2.b"])
        hid = ln2 @ p[f"h.{i}.mlp.w1"] + p[f"h.{i}.mlp.b1"]
        h_act = _gelu(hid)
        mlp = h_act @ p[f"h.{i}.mlp.w2"] + p[f"h.{i}.mlp.b2"]
        out = x2 + mlp
        cache = {
            "ln1": ln1, "ln1c": ln1c, "q": q, "k": k, "v": v,
            "prob": prob, "y": y, "x2": x2, "ln2": ln2, "ln2c": ln2c,
            "h": hid, "h_act": h_act,
        }
        return out, cache

    def forward(self, idx: np.ndarray) -> tuple[np.ndarray, dict]:
        c = self.config
        p = self.params
        t = idx.shape[0]
        tok = p["tok_emb"][idx]
        pos = p["pos_emb"][:t]
        x = tok + pos
        mask = self._mask(t)
        caches = []
        for i in range(c.n_layer):
            x, cache = self._block_forward(x, i, mask)
            caches.append(cache)
        xn, lnc = self._ln(x, p["ln_f.w"], p["ln_f.b"])
        logits = xn @ p["tok_emb"].T
        return logits, {"idx": idx, "xn": xn, "lnc": lnc, "blocks": caches}

    # ------------------------------------------------------------------ KV-cached infer
    def _block_infer(
        self, x: np.ndarray, i: int, past: tuple[np.ndarray, np.ndarray] | None
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        p = self.params
        c = self.config
        t, d = x.shape[0], x.shape[1]
        h = c.n_head
        dh = d // h
        ln1, _ = self._ln(x, p[f"h.{i}.ln1.w"], p[f"h.{i}.ln1.b"])
        qkv = (ln1 @ p[f"h.{i}.attn.wqkv"]).reshape(t, 3, h, dh)
        q = np.transpose(qkv[:, 0], (1, 0, 2))
        k = np.transpose(qkv[:, 1], (1, 0, 2))
        v = np.transpose(qkv[:, 2], (1, 0, 2))
        if past is not None:
            k = np.concatenate([past[0], k], axis=1)
            v = np.concatenate([past[1], v], axis=1)
        att = (q @ np.transpose(k, (0, 2, 1))) * np.float32(1.0 / math.sqrt(dh))
        if t > 1:
            t_all = k.shape[1]
            # causal over the *new* rows vs all keys; past keys are always visible
            past_len = t_all - t
            mask = np.triu(np.full((t, t_all), _NEG_INF, dtype=np.float32), k=1 + past_len)
            att = att + mask
        y = _softmax(att, axis=-1) @ v
        y = np.transpose(y, (1, 0, 2)).reshape(t, d)
        x2 = x + y @ p[f"h.{i}.attn.wo"]
        ln2, _ = self._ln(x2, p[f"h.{i}.ln2.w"], p[f"h.{i}.ln2.b"])
        hid = _gelu(ln2 @ p[f"h.{i}.mlp.w1"] + p[f"h.{i}.mlp.b1"])
        xout = x2 + hid @ p[f"h.{i}.mlp.w2"] + p[f"h.{i}.mlp.b2"]
        return xout, (k, v)

    def forward_infer(
        self, idx: np.ndarray, start_pos: int, past: list | None
    ) -> tuple[np.ndarray, list]:
        p = self.params
        c = self.config
        t = idx.shape[0]
        x = p["tok_emb"][idx] + p["pos_emb"][start_pos : start_pos + t]
        new_past = []
        for i in range(c.n_layer):
            pk = past[i] if past is not None else None
            x, kv = self._block_infer(x, i, pk)
            new_past.append(kv)
        xn, _ = self._ln(x, p["ln_f.w"], p["ln_f.b"])
        logits = xn[-1] @ p["tok_emb"].T
        return logits, new_past

    def generate(
        self,
        prompt: str,
        max_new: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        stop: threading.Event | None = None,
    ) -> str:
        ids = self.tokenizer.encode(prompt)[-self.config.n_ctx :]
        if not ids:
            ids = [self.tokenizer.bos_id]
        out: list[int] = []
        with self.lock:
            ctx = np.asarray(ids, dtype=np.int32)
            logits, past = self.forward_infer(ctx, 0, None)
            for _ in range(max_new):
                if stop is not None and stop.is_set():
                    break
                nxt = _sample_token(logits, temperature, top_k, top_p)
                if nxt == self.tokenizer.eos_id:
                    break
                ids.append(nxt)
                out.append(nxt)
                if len(out) > 8 and len(set(out[-8:])) == 1:
                    break
                if len(ids) > self.config.n_ctx:
                    ids = ids[-self.config.n_ctx :]
                    logits, past = self.forward_infer(np.asarray(ids, dtype=np.int32), 0, None)
                    continue
                logits, past = self.forward_infer(
                    np.asarray([nxt], dtype=np.int32), len(ids) - 1, past
                )
        return self.tokenizer.decode(out)

    # ------------------------------------------------------------------ train
    def train_step(self, text: str, lr: float = CORTEX_LR) -> float:
        ids = self.tokenizer.encode(text)
        if len(ids) < 4:
            return 0.0
        ctx = self.config.n_ctx
        if len(ids) > ctx + 1:
            start = int(np.random.randint(0, len(ids) - ctx - 1))
            ids = ids[start : start + ctx + 1]
        x = np.asarray(ids[:-1], dtype=np.int32)
        y = np.asarray(ids[1:], dtype=np.int32)
        with self.lock:
            loss = self._adam(x, y, lr)
        self.steps += 1
        self.loss_history.append(float(loss))
        if len(self.loss_history) > 400:
            self.loss_history = self.loss_history[-400:]
        return float(loss)

    def _adam(self, idx: np.ndarray, targets: np.ndarray, lr: float) -> float:
        logits, cache = self.forward(idx)
        t, _vsz = logits.shape
        probs = _softmax(logits, axis=-1)
        nll = -np.log(probs[np.arange(t), targets] + 1e-9)
        loss = float(nll.mean())

        dlogits = probs
        dlogits[np.arange(t), targets] -= 1.0
        dlogits *= np.float32(1.0 / t)

        p = self.params
        grads = self._grad_buf()

        xn = cache["xn"]
        grads["tok_emb"] += dlogits.T @ xn
        dxn = dlogits @ p["tok_emb"]
        dx, dw, db = self._ln_backward(dxn, cache["lnc"])
        grads["ln_f.w"] += dw
        grads["ln_f.b"] += db

        c = self.config
        h = c.n_head
        d = c.n_embd
        dh = d // h

        for i in reversed(range(c.n_layer)):
            b = cache["blocks"][i]
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
            dy_h = dy.reshape(t, h, dh).transpose(1, 0, 2)
            dprob = dy_h @ np.transpose(b["v"], (0, 2, 1))
            dv = np.transpose(b["prob"], (0, 2, 1)) @ dy_h
            sp = b["prob"]
            datt = sp * (dprob - (dprob * sp).sum(axis=-1, keepdims=True))
            datt *= np.float32(1.0 / math.sqrt(dh))
            dq = datt @ b["k"]
            dk = np.transpose(datt, (0, 2, 1)) @ b["q"]
            dq_t = dq.transpose(1, 0, 2)
            dk_t = dk.transpose(1, 0, 2)
            dv_t = dv.transpose(1, 0, 2)
            dqkv = np.stack([dq_t, dk_t, dv_t], axis=1).reshape(t, 3 * d)
            grads[f"h.{i}.attn.wqkv"] += b["ln1"].T @ dqkv
            dln1 = dqkv @ p[f"h.{i}.attn.wqkv"].T
            dln1x, dw, db = self._ln_backward(dln1, b["ln1c"])
            grads[f"h.{i}.ln1.w"] += dw
            grads[f"h.{i}.ln1.b"] += db
            dx = dx + dln1x

        np.add.at(grads["tok_emb"], idx, dx)
        grads["pos_emb"][:t] += dx

        # global clip
        norm = math.sqrt(sum(float(np.square(g).sum()) for g in grads.values()) + 1e-12)
        scale = 1.0 if norm < 1.0 else 1.0 / norm

        m, v = self._moments()
        b1, b2, eps = 0.9, 0.999, 1e-8
        # steps is incremented after this call; use steps+1 for bias correction
        tstep = self.steps + 1
        lr_t = lr * math.sqrt(1.0 - b2**tstep) / (1.0 - b1**tstep)
        for k, g in grads.items():
            g *= scale
            m[k] *= b1
            m[k] += (1.0 - b1) * g
            v[k] *= b2
            v[k] += (1.0 - b2) * (g * g)
            p[k] -= (lr_t * m[k] / (np.sqrt(v[k]) + eps)).astype(np.float32)
        return loss

    # ------------------------------------------------------------------ io
    def save(self, path: Path | None = None) -> Path:
        path = path or SAFETENSORS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in self.params.items()}
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
        for pth in (cfg_path, alt_cfg, path.parent / "config.json"):
            if pth.exists():
                try:
                    raw = json.loads(pth.read_text(encoding="utf-8"))
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
        if tensors is None or "tok_emb" not in tensors:
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
