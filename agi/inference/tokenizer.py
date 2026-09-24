"""Byte-level tokenizer with optional HuggingFace tokenizer.json support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

PAD, BOS, EOS = 256, 257, 258
BYTE_VOCAB = 259


class ByteTokenizer:
    """UTF-8 bytes plus three specials. Always available, never fails to encode."""

    pad_id = PAD
    bos_id = BOS
    eos_id = EOS
    vocab_size = BYTE_VOCAB

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i < 256:
                buf.append(i)
        return buf.decode("utf-8", errors="replace")


class HFTokenizer:
    """Minimal BPE / WordPiece reader for HuggingFace tokenizer.json files."""

    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", {})
        self.vocab: dict[str, int] = model.get("vocab") or {}
        self.id_to_token = {int(v): k for k, v in self.vocab.items()}
        merges = model.get("merges") or []
        self.merges: list[tuple[str, str]] = []
        for m in merges:
            if isinstance(m, str):
                parts = m.split()
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                self.merges.append((str(m[0]), str(m[1])))
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.unk = model.get("unk_token") or data.get("unk_token") or "<unk>"
        added = data.get("added_tokens") or []
        for tok in added:
            if isinstance(tok, dict) and "content" in tok and "id" in tok:
                self.vocab[tok["content"]] = int(tok["id"])
                self.id_to_token[int(tok["id"])] = tok["content"]
        self.vocab_size = max(self.vocab.values()) + 1 if self.vocab else BYTE_VOCAB
        self.bos_id = self.vocab.get("<s>", self.vocab.get("<|begin_of_text|>", self.vocab.get("BOS", 1)))
        self.eos_id = self.vocab.get("</s>", self.vocab.get("<|end_of_text|>", self.vocab.get("EOS", 2)))
        self.pad_id = self.vocab.get("<pad>", self.vocab.get("[PAD]", 0))
        pre = data.get("pre_tokenizer") or {}
        self.byte_level = pre.get("type") == "ByteLevel" or model.get("type") == "BPE"

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        if not text:
            ids: list[int] = []
        elif self.merges:
            ids = self._bpe_encode(text)
        else:
            ids = [self.vocab.get(ch, self.vocab.get(self.unk, 0)) for ch in text]
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def _bpe_encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for word in text.split(" "):
            if not word and not tokens:
                continue
            piece = ("Ġ" + word) if tokens or text.startswith(" ") else word
            chars = list(piece) if piece else ["Ġ"]
            if len(chars) == 1:
                tokens.append(self.vocab.get(chars[0], self.vocab.get(self.unk, 0)))
                continue
            pairs = {(chars[i], chars[i + 1]) for i in range(len(chars) - 1)}
            while pairs:
                ranked = min(pairs, key=lambda p: self.ranks.get(p, 10**9))
                if ranked not in self.ranks:
                    break
                a, b = ranked
                new: list[str] = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        new.append(a + b)
                        i += 2
                    else:
                        new.append(chars[i])
                        i += 1
                chars = new
                pairs = {(chars[i], chars[i + 1]) for i in range(len(chars) - 1)}
            for c in chars:
                tokens.append(self.vocab.get(c, self.vocab.get(self.unk, 0)))
            # restore spaces between words for non-byte-level
            if not self.byte_level:
                sp = self.vocab.get(" ", None)
                if sp is not None:
                    tokens.append(sp)
        return tokens

    def decode(self, ids: Iterable[int]) -> str:
        parts = []
        for i in ids:
            tok = self.id_to_token.get(int(i), "")
            if tok in ("<s>", "</s>", "<pad>", "<unk>", "<|begin_of_text|>", "<|end_of_text|>"):
                continue
            parts.append(tok)
        text = "".join(parts)
        text = text.replace("Ġ", " ").replace("▁", " ")
        return text


def load_tokenizer(model_dir: Path) -> ByteTokenizer | HFTokenizer:
    path = model_dir / "tokenizer.json"
    if path.exists():
        try:
            return HFTokenizer(path)
        except Exception:
            pass
    return ByteTokenizer()


def ids_to_array(ids: list[int], ctx: int, pad: int) -> np.ndarray:
    if len(ids) > ctx:
        ids = ids[-ctx:]
    arr = np.full((ctx,), pad, dtype=np.int32)
    arr[: len(ids)] = np.asarray(ids, dtype=np.int32)
    return arr
