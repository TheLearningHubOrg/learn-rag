"""Context assembly: fit the best evidence into a token budget."""

from __future__ import annotations

from app.pipeline import Hit
from app.text import approx_tokens

# Tokens spent on the <document id="..."> wrapper around each chunk.
WRAPPER_TOKENS = 8


def chunk_cost(hit: Hit) -> int:
    """Approximate tokens one hit occupies once wrapped for the prompt."""
    return approx_tokens(hit.chunk.text) + WRAPPER_TOKENS


def pack_context(hits: list[Hit], budget_tokens: int, max_per_doc: int | None = None) -> list[Hit]:
    """Greedy best-first packing.

    Walks the hits in rank order and keeps each one that still fits. A hit
    that does not fit is skipped, not treated as the end, so a short chunk
    further down can still use the remaining space. `max_per_doc` stops one
    long document from filling the whole window.
    """
    if budget_tokens <= 0:
        return []
    packed: list[Hit] = []
    used = 0
    per_doc: dict[str, int] = {}
    for hit in hits:
        cost = chunk_cost(hit)
        if used + cost > budget_tokens:
            continue
        doc = hit.chunk.doc_id
        if max_per_doc is not None and per_doc.get(doc, 0) >= max_per_doc:
            continue
        packed.append(hit)
        used += cost
        per_doc[doc] = per_doc.get(doc, 0) + 1
    return packed
