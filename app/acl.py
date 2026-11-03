"""Document-level access control applied inside retrieval, not after generation."""

from __future__ import annotations

from collections.abc import Iterable

from app.chunking import Chunk
from app.vector_store import Filter


def chunk_groups(chunk: Chunk) -> frozenset[str]:
    """Groups allowed to read a chunk, from its comma-separated `acl` metadata."""
    raw = chunk.metadata.get("acl", "")
    return frozenset(g.strip() for g in raw.split(",") if g.strip())


def allowed_for(groups: Iterable[str]) -> Filter:
    """Build a retrieval filter for a caller who belongs to `groups`.

    Deny by default: a chunk with no `acl` metadata is visible to nobody, so a
    document ingested without labels cannot leak. Pass the result as
    `where=` to `RagPipeline.search`, which applies it to both the dense and
    the BM25 branch.
    """
    caller = frozenset(groups)

    def predicate(chunk: Chunk) -> bool:
        return bool(caller & chunk_groups(chunk))

    return predicate
