"""
The Learning Hub - Retrieval-Augmented Generation (RAG) Pedagogical Mastery Roadmap.
7-Stage progression covering chunking, dense/sparse hybrid search, RRF, reranking, and CRAG/Self-RAG.
"""

from typing import List, Dict, Any

RAG_STAGES: List[Dict[str, Any]] = [
    {
        "stage_number": 1,
        "title": "Absolute Beginner Foundations & RAG Architecture",
        "domain_focus": "Naive RAG vs Modular RAG, parametric vs non-parametric memory, and vector embeddings.",
        "completed": True,
        "key_topics": ["Naive RAG Bottlenecks", "Parametric Knowledge Limits", "Vector Similarity (Cosine/Dot)", "Modular RAG Triad"]
    },
    {
        "stage_number": 2,
        "title": "Document Ingestion, Parsing & Chunking Strategies",
        "domain_focus": "Recursive token splitting, semantic chunking, layout-aware PDF parsing, and metadata extraction.",
        "completed": True,
        "key_topics": ["Fixed-size vs Semantic Chunking", "Chunk Overlap & Sliding Windows", "Parent Document Chunking", "Metadata Injection"]
    },
    {
        "stage_number": 3,
        "title": "Embeddings, Dense vs Sparse Retrieval & Hybrid Search",
        "domain_focus": "Bi-encoder dense models, BM25/SPLADE sparse models, and Reciprocal Rank Fusion (RRF).",
        "completed": True,
        "key_topics": ["Dense vs Sparse Trade-offs", "BM25 Term Frequency & Inverse Doc Freq", "Reciprocal Rank Fusion (k=60)", "Vector Databases"]
    },
    {
        "stage_number": 4,
        "title": "Advanced Retrieval, Query Transformation & Reranking",
        "domain_focus": "Multi-query expansion, HyDE (Hypothetical Document Embeddings), and Cross-Encoder reranking.",
        "completed": True,
        "key_topics": ["Query Rewriting & Sub-Queries", "HyDE Hallucinated Drafts", "Cross-Encoder Joint Attention", "Top-K Filtering"]
    },
    {
        "stage_number": 5,
        "title": "Context Injection, Compression & Prompt Engineering",
        "domain_focus": "Long-context lost in the middle mitigation, contextual compression, and citation verification.",
        "completed": True,
        "key_topics": ["Lost-in-the-Middle Phenomenon", "LLMLingua Context Compression", "Exact Source Grounding", "Strict Schema Prompts"]
    },
    {
        "stage_number": 6,
        "title": "Corrective RAG (CRAG), Self-RAG & Agentic RAG",
        "domain_focus": "Document relevance grading, corrective web fallback, reflection tokens ([IsRel], [IsSup]), and cyclic loops.",
        "completed": True,
        "key_topics": ["CRAG Relevance Evaluator", "Fallback Web Knowledge Retrieval", "Self-RAG Reflection Tokens", "Adaptive Router RAG"]
    },
    {
        "stage_number": 7,
        "title": "Staff RAG Architect: Enterprise Blueprint, Evaluation & 50 Interview Q&As",
        "domain_focus": "Ragas automated evaluation, production latency budgets, disaster recovery, and 50 staff Q&As.",
        "completed": True,
        "key_topics": ["Ragas Faithfulness & Context Recall", "P99 SLA & Multi-Stage Caching", "Zero-Downtime Re-indexing", "Enterprise Cheatsheet"]
    }
]

def get_completed_count() -> int:
    return sum(1 for s in RAG_STAGES if s["completed"])

def get_progress_percentage() -> int:
    return round((get_completed_count() / len(RAG_STAGES)) * 100)
