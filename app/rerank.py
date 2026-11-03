"""Second-stage reranking: rescore a shortlist with a scorer that reads query and passage together."""

from __future__ import annotations

from typing import Protocol

from app.pipeline import Hit
from app.text import tokenize


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class OverlapReranker:
    """Dependency-free stand-in for a cross-encoder.

    It scores each passage by query-term coverage (70%) plus the share of the
    query's adjacent word pairs that appear in the same order in the passage
    (30%). Like a cross-encoder it looks at the query and the passage
    together, so word order counts; unlike one, it has no notion of meaning.
    Swap in `sentence_transformers.CrossEncoder` behind the same protocol.
    """

    def score(self, query: str, passages: list[str]) -> list[float]:
        terms = tokenize(query)
        if not terms:
            return [0.0] * len(passages)
        wanted = set(terms)
        pairs = set(zip(terms, terms[1:], strict=False))
        scores: list[float] = []
        for passage in passages:
            tokens = tokenize(passage)
            coverage = len(wanted & set(tokens)) / len(wanted)
            seen_pairs = set(zip(tokens, tokens[1:], strict=False))
            phrase = len(pairs & seen_pairs) / len(pairs) if pairs else 0.0
            scores.append(round(0.7 * coverage + 0.3 * phrase, 6))
        return scores


def rerank(
    query: str,
    hits: list[Hit],
    reranker: Reranker,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[Hit]:
    """Rescore `hits`, drop those under `min_score`, keep the best `top_k`.

    The returned hits carry the reranker score; the retrieval ranks are kept
    so you can still see where each hit came from.
    """
    scores = reranker.score(query, [h.chunk.text for h in hits])
    ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: (-pair[1], pair[0].chunk.chunk_id))
    kept = [
        Hit(hit.chunk, score, hit.dense_rank, hit.sparse_rank)
        for hit, score in ranked
        if min_score is None or score >= min_score
    ]
    return kept[:top_k]
