"""GGUF reader + optional llama.cpp inference.

The custom server always parses `model/model.gguf` itself so the UI can show
architecture, parameter count, and quantization even when llama.cpp is absent.
Generation uses llama-cpp-python when installed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

GGUF_MAGIC = b"GGUF"

GGUF_T = {
    0: "u8",
    1: "i8",
    2: "u16",
    3: "i16",
    4: "u32",
    5: "i32",
    6: "f32",
    7: "bool",
    8: "string",
    9: "array",
    10: "u64",
    11: "i64",
    12: "f64",
}

_UNPACK = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<B", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}


class _R:
    def __init__(self, f):
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated GGUF")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, typ: int) -> Any:
        if typ == 8:
            return self.string()
        if typ == 7:
            return bool(self.read(1)[0])
        if typ == 9:
            at = self.u32()
            n = self.u64()
            # skip large arrays but keep short ones (architecture strings, etc.)
            if n > 64 or at == 9:
                # consume
                for _ in range(n):
                    self.value(at)
                return f"<array {GGUF_T.get(at, at)} x {n}>"
            return [self.value(at) for _ in range(n)]
        fmt, sz = _UNPACK[typ]
        return struct.unpack(fmt, self.read(sz))[0]


@dataclass
class GGUFInfo:
    path: str
    version: int
    n_tensors: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[dict] = field(default_factory=list)

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture") or "unknown")

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name") or Path(self.path).name)

    @property
    def context_length(self) -> int | None:
        arch = self.architecture
        key = f"{arch}.context_length"
        v = self.metadata.get(key)
        return int(v) if v is not None else None

    @property
    def block_count(self) -> int | None:
        arch = self.architecture
        v = self.metadata.get(f"{arch}.block_count")
        return int(v) if v is not None else None

    @property
    def embedding_length(self) -> int | None:
        arch = self.architecture
        v = self.metadata.get(f"{arch}.embedding_length")
        return int(v) if v is not None else None

    def summary(self) -> dict:
        quant = None
        if self.tensors:
            quant = self.tensors[0].get("dtype")
        return {
            "path": self.path,
            "version": self.version,
            "name": self.name,
            "architecture": self.architecture,
            "n_tensors": self.n_tensors,
            "block_count": self.block_count,
            "embedding_length": self.embedding_length,
            "context_length": self.context_length,
            "quant": quant,
            "metadata_keys": list(self.metadata.keys())[:40],
        }


def parse_gguf(path: Path, max_tensors: int = 32) -> GGUFInfo:
    with open(path, "rb") as fh:
        r = _R(fh)
        magic = r.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic={magic!r})")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            typ = r.u32()
            try:
                meta[key] = r.value(typ)
            except Exception:
                break
        tensors = []
        try:
            for i in range(min(int(n_tensors), max_tensors)):
                name = r.string()
                n_dims = r.u32()
                dims = [r.u64() for _ in range(n_dims)]
                dtype = r.u32()
                offset = r.u64()
                tensors.append({"name": name, "dims": dims, "dtype": dtype, "offset": offset})
        except Exception:
            pass
        return GGUFInfo(str(path), version, int(n_tensors), meta, tensors)


class LlamaCppEngine:
    def __init__(self, path: Path, info: GGUFInfo):
        from llama_cpp import Llama

        n_ctx = min(info.context_length or 2048, 2048)
        self.llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=2,
            verbose=False,
        )
        self.info = info
        self.path = path

    def generate(self, prompt: str, max_new: int = 128, temperature: float = 0.8, top_k: int = 40) -> str:
        out = self.llm(
            prompt,
            max_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            stop=["</s>", "<|eot_id|>", "<|im_end|>", "<|user|>", "\nUser:", "\nUSER:"],
        )
        return out["choices"][0]["text"]

    def stream(self, prompt: str, max_new: int = 128, temperature: float = 0.8, top_k: int = 40) -> Iterator[str]:
        for chunk in self.llm(
            prompt,
            max_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            stream=True,
            stop=["</s>", "<|eot_id|>", "<|im_end|>", "<|user|>"],
        ):
            text = chunk["choices"][0]["text"]
            if text:
                yield text
