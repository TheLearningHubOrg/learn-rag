"""Combine ranked lists from different retrievers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

Ranking = Sequence[tuple[str, float]]


def reciprocal_rank_fusion(rankings: Sequence[Ranking], k: int = 60) -> list[tuple[str, float]]:
    """RRF: score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Raw scores are ignored, so BM25 (unbounded) and cosine ([-1, 1]) can be
    fused without calibration. Documents missing from a list get nothing
    from that list.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def min_max(ranking: Ranking) -> dict[str, float]:
    if not ranking:
        return {}
    scores = [s for _, s in ranking]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {d: 1.0 for d, _ in ranking}
    return {d: (s - lo) / (hi - lo) for d, s in ranking}


def convex_combination(dense: Ranking, sparse: Ranking, alpha: float = 0.5) -> list[tuple[str, float]]:
    """alpha * dense + (1 - alpha) * sparse, after min-max scaling each list to [0, 1]."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    d, s = min_max(dense), min_max(sparse)
    ids = set(d) | set(s)
    combined = {i: alpha * d.get(i, 0.0) + (1 - alpha) * s.get(i, 0.0) for i in ids}
    return sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
