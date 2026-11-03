"""Semantic answer cache keyed by query embedding and corpus version."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from app.embeddings import Embedder


@dataclass
class _Entry:
    vector: list[float]
    answer: str
    corpus_version: str
    expires_at: float


class SemanticCache:
    """Return a stored answer when a new query is close enough to an old one.

    Three things must match: the cosine similarity of the query vectors is at
    least `threshold`, the corpus version is the same (so an answer never
    outlives the documents it came from), and the entry has not expired.
    Eviction is least-recently-used. The scan is linear, which is fine for a
    few thousand entries; past that, put the vectors in an ANN index.
    """

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.92,
        ttl_seconds: float = 3600.0,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.embedder = embedder
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, query: str, corpus_version: str) -> str | None:
        now = self.clock()
        vec = self.embedder.embed([query])[0]
        best_key, best_sim = None, -1.0
        for key, entry in list(self._entries.items()):
            if entry.expires_at <= now:
                del self._entries[key]
                continue
            if entry.corpus_version != corpus_version:
                continue
            sim = sum(a * b for a, b in zip(vec, entry.vector, strict=True))  # unit vectors
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_key is None or best_sim < self.threshold:
            return None
        self._entries.move_to_end(best_key)
        return self._entries[best_key].answer

    def put(self, query: str, answer: str, corpus_version: str) -> None:
        key = " ".join(query.lower().split())
        vec = self.embedder.embed([query])[0]
        self._entries[key] = _Entry(vec, answer, corpus_version, self.clock() + self.ttl)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
