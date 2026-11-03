"""Exact (brute-force) vector search with metadata pre-filtering."""

from __future__ import annotations

from collections.abc import Callable

from app.chunking import Chunk

Filter = Callable[[Chunk], bool]


class InMemoryVectorStore:
    """Stores unit vectors, so the dot product equals cosine similarity.

    Exact search is O(n * d) per query. That is fine up to roughly 100k
    vectors on one core; past that you want an ANN index (HNSW, IVF).
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        self._chunks[chunk.chunk_id] = chunk
        self._vectors[chunk.chunk_id] = vector

    def delete_document(self, doc_id: str) -> int:
        ids = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in ids:
            del self._chunks[cid]
            del self._vectors[cid]
        return len(ids)

    def get(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def search(
        self, query_vec: list[float], top_k: int = 10, where: Filter | None = None
    ) -> list[tuple[str, float]]:
        scored = []
        for cid, vec in self._vectors.items():
            if where is not None and not where(self._chunks[cid]):
                continue  # pre-filter: excluded chunks are never scored
            scored.append((cid, sum(a * b for a, b in zip(query_vec, vec, strict=True))))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]
