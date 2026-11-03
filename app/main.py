"""FastAPI service exposing ingestion, hybrid search and cited answers."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.corpus import SEED_DOCUMENTS
from app.pipeline import RagPipeline


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    markdown: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class HitOut(BaseModel):
    chunk_id: str
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    text: str


def create_app(seed: bool = True) -> FastAPI:
    app = FastAPI(title="learn-rag starter", version="1.0.0")
    pipeline = RagPipeline()
    if seed:
        for doc_id, text in SEED_DOCUMENTS.items():
            pipeline.ingest(doc_id, text)
    app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "chunks": len(pipeline.store)}

    @app.post("/documents", status_code=201)
    def ingest(doc: DocumentIn) -> dict[str, object]:
        written = pipeline.ingest(doc.doc_id, doc.markdown, doc.metadata)
        return {"doc_id": doc.doc_id, "chunks_written": written}

    @app.delete("/documents/{doc_id}")
    def delete(doc_id: str) -> dict[str, object]:
        removed = pipeline.delete(doc_id)
        if not removed:
            raise HTTPException(status_code=404, detail="unknown document")
        return {"doc_id": doc_id, "chunks_removed": removed}

    @app.post("/search")
    def search(q: Query) -> list[HitOut]:
        return [
            HitOut(
                chunk_id=h.chunk.chunk_id,
                score=round(h.score, 6),
                dense_rank=h.dense_rank,
                sparse_rank=h.sparse_rank,
                text=h.chunk.text,
            )
            for h in pipeline.search(q.query, q.top_k)
        ]

    @app.post("/ask")
    def ask(q: Query) -> dict[str, object]:
        result = pipeline.answer(q.query, min(q.top_k, 5))
        return {
            "question": result.question,
            "grade": result.grade,
            "answer": result.answer,
            "citations": result.citations,
        }

    return app


app = create_app()
