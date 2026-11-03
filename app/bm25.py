"""Okapi BM25 over an in-memory inverted index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.text import tokenize


class BM25Index:
    """Sparse lexical index.

    k1 controls term-frequency saturation, b controls document-length
    normalisation. The IDF uses the "+1 inside the log" variant, so a term
    that appears in every document still scores slightly above zero instead
    of going negative.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._lengths)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._lengths:
            self.remove(doc_id)
        tokens = tokenize(text)
        self._lengths[doc_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            self._postings[term][doc_id] = tf

    def remove(self, doc_id: str) -> None:
        if self._lengths.pop(doc_id, None) is None:
            return
        for term in list(self._postings):
            self._postings[term].pop(doc_id, None)
            if not self._postings[term]:
                del self._postings[term]

    def idf(self, term: str) -> float:
        n_docs = len(self._lengths)
        df = len(self._postings.get(term, {}))
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._lengths:
            return []
        avgdl = sum(self._lengths.values()) / len(self._lengths)
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings.items():
                norm = 1 - self.b + self.b * self._lengths[doc_id] / avgdl
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]
