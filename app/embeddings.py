"""Embedding interface plus a dependency-free stand-in model.

`HashingEmbedder` is NOT a neural model. It hashes word unigrams and
character trigrams into a fixed number of dimensions and L2-normalises the
result. That gives it two properties the guide relies on: it is deterministic
(tests can assert exact rankings) and it rewards partial word overlap
("embedding" vs "embeddings"), which BM25 does not. It does not know that
"car" and "automobile" are related; a real bi-encoder does. Swap it for a
sentence-transformers or API model by implementing the same `Embedder`
protocol.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from app.text import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def truncate(vec: list[float], dim: int) -> list[float]:
    """Matryoshka-style truncation: keep the first `dim` values, then re-normalise."""
    return l2_normalize(vec[:dim])


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0  # signed hashing reduces collision bias
        return (value >> 1) % self.dim, sign

    def _features(self, text: str) -> list[tuple[str, float]]:
        feats: list[tuple[str, float]] = []
        for token in tokenize(text):
            feats.append((f"w:{token}", 1.0))
            padded = f"^{token}$"
            for i in range(len(padded) - 2):
                feats.append((f"c:{padded[i : i + 3]}", 0.5))
        return feats

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for feature, weight in self._features(text):
                idx, sign = self._bucket(feature)
                vec[idx] += sign * weight
            out.append(l2_normalize(vec))
        return out
