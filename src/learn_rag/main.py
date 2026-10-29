"""
The Learning Hub - Retrieval-Augmented Generation (RAG) Architecture CLI.
Demonstrates Hybrid Search, RRF ranking, Cross-Encoder reranking, CRAG fallbacks, and Self-RAG reflection tokens.
"""

from __future__ import annotations
import sys
from .roadmap import RAG_STAGES, get_completed_count, get_progress_percentage
from .rag import (
    DocumentChunk,
    HybridRetriever,
    CrossEncoderReranker,
    RAGPipeline
)


def print_banner() -> None:
    print("=" * 76)
    print("   THE LEARNING HUB: RETRIEVAL-AUGMENTED GENERATION (RAG) ENGINE   ")
    print("=" * 76)


def display_roadmap() -> None:
    print("\n[+] 7-Stage RAG Pedagogical Mastery Roadmap:")
    for stage in RAG_STAGES:
        status = "[DONE]" if stage["completed"] else "[TODO]"
        print(f"  {status} Stage {stage['stage_number']}: {stage['title']}")
        print(f"         Focus: {stage['domain_focus']}")
    print(f"\nProgress: {get_completed_count()}/{len(RAG_STAGES)} Stages Complete ({get_progress_percentage()}%)\n")


def build_sample_knowledge_base() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text="Reciprocal Rank Fusion (RRF) with constant k=60 fuses disparate ranking lists from dense "
                 "vector embeddings and sparse BM25 models without requiring calibrated score normalization.",
            metadata={"topic": "hybrid_search", "doc_id": "doc_rrf"}
        ),
        DocumentChunk(
            text="Cross-Encoder models apply joint all-to-all attention across query and document pairs, "
                 "yielding significantly higher ranking precision compared to bi-encoder cosine similarity.",
            metadata={"topic": "reranking", "doc_id": "doc_cross_encoder"}
        ),
        DocumentChunk(
            text="Corrective RAG (CRAG) evaluates retrieved document relevance and triggers fallback web "
                 "search when knowledge base documents fail to meet relevance thresholds.",
            metadata={"topic": "crag", "doc_id": "doc_crag"}
        ),
        DocumentChunk(
            text="Self-RAG dynamically emits special reflection tokens like [Retrieve], [IsRel], and [IsSup] "
                 "to evaluate factual consistency and ground generation in verifiable context.",
            metadata={"topic": "self_rag", "doc_id": "doc_self_rag"}
        ),
    ]


def demonstrate_hybrid_search_and_rrf(docs: list[DocumentChunk]) -> None:
    print("-" * 68)
    print("1. Hybrid Retrieval (Dense Vector + Sparse BM25) & RRF (k=60)")
    print("-" * 68)

    retriever = HybridRetriever(rrf_k=60)
    retriever.add_chunks(docs)

    query = "How does Reciprocal Rank Fusion combine dense and sparse rankings?"
    print(f"  [Query] '{query}'")
    top_chunks = retriever.retrieve(query, top_k=2)

    print(f"  [+] Retrieved {len(top_chunks)} chunks via RRF:")
    for i, chunk in enumerate(top_chunks):
        print(f"    Rank {i+1} [RRF Score: {chunk.score:.5f}] -> '{chunk.text[:75]}...'")
    print()


def demonstrate_crag_and_self_rag(docs: list[DocumentChunk]) -> None:
    print("-" * 68)
    print("2. Corrective RAG (CRAG) & Self-RAG Dynamic Reflection")
    print("-" * 68)

    retriever = HybridRetriever(rrf_k=60)
    retriever.add_chunks(docs)
    pipeline = RAGPipeline(retriever=retriever)

    # Scenario 1: High Relevance Query
    q1 = "Explain Cross-Encoder joint attention reranking"
    res1 = pipeline.query(q1)
    print(f"  [Scenario 1: In-Domain Query] '{q1}'")
    print(f"    CRAG Grade:     {res1['crag_grade']} (Confidence: {res1['confidence']})")
    print(f"    Reflection:     {res1['reflection_tokens']['is_rel']} | {res1['reflection_tokens']['is_sup']}")
    print(f"    Synthesized Answer:\n      {res1['response'].splitlines()[-1]}")

    # Scenario 2: Out-of-Domain Query triggering CRAG Fallback
    q2 = "What are the quantum physics equations for warp drives?"
    res2 = pipeline.query(q2)
    print(f"\n  [Scenario 2: Out-of-Domain Query] '{q2}'")
    print(f"    CRAG Grade:     {res2['crag_grade']} (Confidence: {res2['confidence']})")
    print(f"    Fallback Action: Discarded irrelevant local chunks -> Triggered Fallback Web Search!")
    print(f"    Chunks Used:    {res2['chunks_used']}")
    print(f"    Synthesized Answer:\n      {res2['response'].splitlines()[-1]}\n")


def main() -> None:
    print_banner()
    display_roadmap()
    docs = build_sample_knowledge_base()
    demonstrate_hybrid_search_and_rrf(docs)
    demonstrate_crag_and_self_rag(docs)
    print("=" * 76)
    print("   RAG ARCHITECTURE PRODUCTION ENGINE RUN COMPLETED   ")
    print("=" * 76)


if __name__ == "__main__":
    main()
