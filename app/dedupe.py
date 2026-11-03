"""Near-duplicate detection with MinHash signatures."""

from __future__ import annotations

import hashlib

from app.text import tokenize

_PRIME = (1 << 61) - 1  # Mersenne prime used for the universal hash family


def shingles(text: str, n: int = 3) -> set[str]:
    """Overlapping n-word sequences: the unit MinHash compares."""
    words = tokenize(text, drop_stopwords=False)
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")


class MinHasher:
    """Signature of `num_perm` minimum hash values.

    For one random hash function, the chance that two sets share the same
    minimum equals their Jaccard similarity. With many functions, the share
    of agreeing positions estimates Jaccard without comparing the sets.
    """

    def __init__(self, num_perm: int = 128, seed: int = 7) -> None:
        self._params = [
            (_hash(f"a:{seed}:{i}") % _PRIME or 1, _hash(f"b:{seed}:{i}") % _PRIME)
            for i in range(num_perm)
        ]

    def signature(self, text: str) -> list[int]:
        hashes = [_hash(s) for s in shingles(text)] or [0]
        return [min((a * h + b) % _PRIME for h in hashes) for a, b in self._params]

    @staticmethod
    def similarity(sig_a: list[int], sig_b: list[int]) -> float:
        return sum(x == y for x, y in zip(sig_a, sig_b, strict=True)) / len(sig_a)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def drop_near_duplicates(texts: list[str], threshold: float = 0.8) -> list[str]:
    """Keep the first copy of each group of near-identical texts.

    This compares every pair, O(n^2). At corpus scale, split signatures into
    LSH bands so only texts that collide in some band are compared.
    """
    hasher = MinHasher()
    kept: list[tuple[str, list[int]]] = []
    for text in texts:
        sig = hasher.signature(text)
        if all(MinHasher.similarity(sig, other) < threshold for _, other in kept):
            kept.append((text, sig))
    return [t for t, _ in kept]
