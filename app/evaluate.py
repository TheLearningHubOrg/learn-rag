"""Compare dense-only, BM25-only and hybrid retrieval on a labelled query set.

Run: python -m app.evaluate
"""

from __future__ import annotations

import statistics

from app.corpus import SEED_DOCUMENTS
from app.fusion import reciprocal_rank_fusion
from app.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from app.pipeline import RagPipeline

# query -> ids of the chunks a human marked as relevant
GOLDEN: dict[str, set[str]] = {
    "how are dense and sparse rankings fused": {"rrf#0"},
    "k=60 constant in rank fusion": {"rrf#0"},
    "CVE-2024-3094": {"bm25#0"},
    "term frequency saturation parameter": {"bm25#0"},
    "heading breadcrumb on each passage": {"chunking#0"},
    "rerank fused candidates before the prompt": {"reranking#0"},
    "cosine distance operator in postgres": {"pgvector#0"},
    "ef_search recall latency": {"pgvector#0"},
}


def run(k: int = 3) -> dict[str, dict[str, float]]:
    pipe = RagPipeline()
    for doc_id, text in SEED_DOCUMENTS.items():
        pipe.ingest(doc_id, text)

    def dense(q: str) -> list[str]:
        return [cid for cid, _ in pipe.store.search(pipe.embedder.embed([q])[0], k)]

    def sparse(q: str) -> list[str]:
        return [cid for cid, _ in pipe.bm25.search(q, k)]

    def hybrid(q: str) -> list[str]:
        d = pipe.store.search(pipe.embedder.embed([q])[0], 20)
        s = pipe.bm25.search(q, 20)
        return [cid for cid, _ in reciprocal_rank_fusion([d, s])[:k]]

    report: dict[str, dict[str, float]] = {}
    for name, retrieve in {"dense": dense, "bm25": sparse, "hybrid": hybrid}.items():
        runs = [(retrieve(q), rel) for q, rel in GOLDEN.items()]
        report[name] = {
            f"recall@{k}": statistics.fmean(recall_at_k(r, rel, k) for r, rel in runs),
            "mrr": statistics.fmean(reciprocal_rank(r, rel) for r, rel in runs),
            f"ndcg@{k}": statistics.fmean(ndcg_at_k(r, rel, k) for r, rel in runs),
        }
    return report


if __name__ == "__main__":
    for name, scores in run().items():
        print(f"{name:7s} " + "  ".join(f"{m}={v:.3f}" for m, v in scores.items()))
