"""Forward-only numpy inference for Llama / GPT-2 / Qwen-style safetensors."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from agi.config import CONFIG_PATH, MAX_NUMPY_BYTES, MAX_NUMPY_PARAMS, MODEL_DIR

from .tokenizer import load_tokenizer


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x, -60, 60))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def _rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps)) * w


def _rope(q: np.ndarray, k: np.ndarray, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """q,k: [T, H, D] with even D."""
    T, _, D = q.shape
    half = D // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    t = np.arange(T, dtype=np.float32)
    ang = np.outer(t, freqs)  # [T, half]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]

    def apply(x):
        a, b = x[..., :half], x[..., half:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)

    return apply(q), apply(k)


def load_tensors(path: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return load_file(str(path))


def detect_arch(keys: list[str]) -> str:
    s = " ".join(keys[:40]) + " " + " ".join(keys)
    if "tok_emb" in keys and any(k.startswith("h.0.attn.wqkv") for k in keys):
        return "cortex-gpt"
    if "model.layers.0.self_attn.q_proj.weight" in keys or "model.layers.0.self_attn.q_proj.weight" in s:
        return "llama"
    if any(k.startswith("h.0.attn.c_attn") or k.startswith("transformer.h.0.attn.c_attn") for k in keys):
        return "gpt2"
    if "model.embed_tokens.weight" in keys:
        return "llama"
    return "unknown"


def tensor_nbytes(tensors: dict[str, np.ndarray]) -> int:
    return int(sum(v.nbytes for v in tensors.values()))


def tensor_params(tensors: dict[str, np.ndarray]) -> int:
    return int(sum(v.size for v in tensors.values()))


class LlamaNumpy:
    def __init__(self, tensors: dict[str, np.ndarray], config: dict, tokenizer):
        self.t = tensors
        self.cfg = config
        self.tokenizer = tokenizer
        self.n_layer = int(config.get("num_hidden_layers") or config.get("n_layer") or _count_layers(tensors))
        self.n_heads = int(config.get("num_attention_heads") or config.get("n_head") or 8)
        hidden = tensors.get("model.embed_tokens.weight")
        self.d = int(config.get("hidden_size") or (hidden.shape[1] if hidden is not None else 256))
        self.n_kv = int(config.get("num_key_value_heads") or self.n_heads)
        self.eps = float(config.get("rms_norm_eps") or 1e-5)
        self.theta = float(config.get("rope_theta") or 10000.0)
        self.head_dim = self.d // self.n_heads

    def _w(self, key: str) -> np.ndarray:
        return self.t[key]

    def generate(self, prompt: str, max_new: int = 80, temperature: float = 0.8, top_k: int = 40) -> str:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [1]
        for _ in range(max_new):
            logits = self.forward(np.asarray(ids, dtype=np.int32))
            logits = logits[-1] / max(temperature, 1e-6)
            if top_k and top_k < logits.shape[0]:
                thresh = np.partition(logits, -top_k)[-top_k]
                logits = np.where(logits < thresh, -1e9, logits)
            probs = _softmax(logits)
            nxt = int(np.random.choice(len(probs), p=probs.astype(np.float64)))
            ids.append(nxt)
            if nxt in (self.tokenizer.eos_id, 0, 2):
                break
        return self.tokenizer.decode(ids[len(self.tokenizer.encode(prompt)) :])

    def forward(self, idx: np.ndarray) -> np.ndarray:
        x = self._w("model.embed_tokens.weight")[idx]
        for i in range(self.n_layer):
            x = self._layer(x, i)
        x = _rms_norm(x, self._w("model.norm.weight"), self.eps)
        w_head = self.t.get("lm_head.weight", self._w("model.embed_tokens.weight"))
        return x @ w_head.T

    def _layer(self, x: np.ndarray, i: int) -> np.ndarray:
        prefix = f"model.layers.{i}"
        h = _rms_norm(x, self._w(f"{prefix}.input_layernorm.weight"), self.eps)
        q = h @ self._w(f"{prefix}.self_attn.q_proj.weight").T
        k = h @ self._w(f"{prefix}.self_attn.k_proj.weight").T
        v = h @ self._w(f"{prefix}.self_attn.v_proj.weight").T
        T = x.shape[0]
        q = q.reshape(T, self.n_heads, self.head_dim)
        k = k.reshape(T, self.n_kv, self.head_dim)
        v = v.reshape(T, self.n_kv, self.head_dim)
        q, k = _rope(q, k, self.theta)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // max(self.n_kv, 1)
            k = np.repeat(k, rep, axis=1)
            v = np.repeat(v, rep, axis=1)
        qh = np.transpose(q, (1, 0, 2))
        kh = np.transpose(k, (1, 0, 2))
        vh = np.transpose(v, (1, 0, 2))
        att = (qh @ np.transpose(kh, (0, 2, 1))) / math.sqrt(self.head_dim)
        att = att + np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
        y = _softmax(att, axis=-1) @ vh
        y = np.transpose(y, (1, 0, 2)).reshape(T, self.d)
        x = x + y @ self._w(f"{prefix}.self_attn.o_proj.weight").T
        h = _rms_norm(x, self._w(f"{prefix}.post_attention_layernorm.weight"), self.eps)
        if f"{prefix}.mlp.gate_proj.weight" in self.t:
            g = _silu(h @ self._w(f"{prefix}.mlp.gate_proj.weight").T)
            u = h @ self._w(f"{prefix}.mlp.up_proj.weight").T
            h2 = (g * u) @ self._w(f"{prefix}.mlp.down_proj.weight").T
        else:
            h2 = _silu(h @ self._w(f"{prefix}.mlp.up_proj.weight").T) @ self._w(f"{prefix}.mlp.down_proj.weight").T
        return x + h2


class GPT2Numpy:
    def __init__(self, tensors: dict[str, np.ndarray], config: dict, tokenizer):
        self.t = {k.replace("transformer.", ""): v for k, v in tensors.items()}
        self.cfg = config
        self.tokenizer = tokenizer
        self.n_layer = int(config.get("n_layer") or _count_gpt2_layers(self.t))
        wte = self.t.get("wte.weight")
        self.d = int(config.get("n_embd") or (wte.shape[1] if wte is not None else 768))
        self.n_head = int(config.get("n_head") or 12)

    def generate(self, prompt: str, max_new: int = 80, temperature: float = 0.8, top_k: int = 40) -> str:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [self.tokenizer.bos_id]
        start = len(ids)
        for _ in range(max_new):
            logits = self.forward(np.asarray(ids[-1024:], dtype=np.int32))
            logits = logits[-1] / max(temperature, 1e-6)
            if top_k and top_k < logits.shape[0]:
                thresh = np.partition(logits, -top_k)[-top_k]
                logits = np.where(logits < thresh, -1e9, logits)
            probs = _softmax(logits)
            nxt = int(np.random.choice(len(probs), p=probs.astype(np.float64)))
            ids.append(nxt)
            if nxt == self.tokenizer.eos_id:
                break
        return self.tokenizer.decode(ids[start:])

    def _ln(self, x, w, b, eps=1e-5):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return w * (x - mean) / np.sqrt(var + eps) + b

    def forward(self, idx: np.ndarray) -> np.ndarray:
        t = self.t
        T = idx.shape[0]
        x = t["wte.weight"][idx] + t["wpe.weight"][:T]
        H = self.n_head
        dh = self.d // H
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
        for i in range(self.n_layer):
            h = self._ln(x, t[f"h.{i}.ln_1.weight"], t[f"h.{i}.ln_1.bias"])
            qkv_w = t[f"h.{i}.attn.c_attn.weight"]
            qkv_b = t.get(f"h.{i}.attn.c_attn.bias")
            qkv = h @ qkv_w + (qkv_b if qkv_b is not None else 0)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, H, dh).transpose(1, 0, 2)
            k = k.reshape(T, H, dh).transpose(1, 0, 2)
            v = v.reshape(T, H, dh).transpose(1, 0, 2)
            att = (q @ np.transpose(k, (0, 2, 1))) / math.sqrt(dh) + mask
            y = _softmax(att, -1) @ v
            y = y.transpose(1, 0, 2).reshape(T, self.d)
            x = x + y @ t[f"h.{i}.attn.c_proj.weight"] + t.get(f"h.{i}.attn.c_proj.bias", 0)
            h = self._ln(x, t[f"h.{i}.ln_2.weight"], t[f"h.{i}.ln_2.bias"])
            h1 = h @ t[f"h.{i}.mlp.c_fc.weight"] + t.get(f"h.{i}.mlp.c_fc.bias", 0)
            h1 = 0.5 * h1 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (h1 + 0.044715 * h1**3)))
            x = x + h1 @ t[f"h.{i}.mlp.c_proj.weight"] + t.get(f"h.{i}.mlp.c_proj.bias", 0)
        x = self._ln(x, t["ln_f.weight"], t["ln_f.bias"])
        return x @ t["wte.weight"].T


def _count_layers(tensors: dict) -> int:
    n = 0
    while f"model.layers.{n}.self_attn.q_proj.weight" in tensors:
        n += 1
    return n


def _count_gpt2_layers(tensors: dict) -> int:
    n = 0
    while f"h.{n}.attn.c_attn.weight" in tensors:
        n += 1
    return n


def read_hf_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def try_load_external_safetensors(path: Path):
    """Return (kind, runner_or_none, info_dict)."""
    try:
        tensors = load_tensors(path)
    except Exception as e:
        return "error", None, {"error": f"failed to read safetensors: {e}"}
    keys = list(tensors.keys())
    arch = detect_arch(keys)
    n_params = tensor_params(tensors)
    n_bytes = tensor_nbytes(tensors)
    info = {
        "arch": arch,
        "tensors": len(keys),
        "params": n_params,
        "bytes": n_bytes,
        "keys_preview": keys[:12],
    }
    if arch == "cortex-gpt":
        return "cortex-gpt", tensors, info
    if n_params > MAX_NUMPY_PARAMS or n_bytes > MAX_NUMPY_BYTES:
        info["error"] = (
            f"checkpoint is too large for the numpy engine ({n_params:,} params). "
            "Convert to GGUF and place it at model/model.gguf, or install llama-cpp-python."
        )
        return "too_large", None, info
    tok = load_tokenizer(MODEL_DIR)
    cfg = read_hf_config()
    try:
        if arch == "llama":
            return "llama", LlamaNumpy(tensors, cfg, tok), info
        if arch == "gpt2":
            return "gpt2", GPT2Numpy(tensors, cfg, tok), info
    except Exception as e:
        info["error"] = str(e)
        return "error", None, info
    info["error"] = f"unknown safetensors layout ({arch})"
    return "unknown", None, info
