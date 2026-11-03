"""Seed knowledge base loaded at start-up so /search and /ask work immediately."""

SEED_DOCUMENTS: dict[str, str] = {
    "rrf": """# Hybrid Search
## Reciprocal Rank Fusion
Reciprocal Rank Fusion combines ranked lists from a dense retriever and a BM25 retriever.
Each document scores 1 / (k + rank) in every list it appears in, with k usually set to 60.
Because only ranks are used, the raw scores never need to be calibrated.
""",
    "bm25": """# Sparse Retrieval
## BM25
BM25 scores a document by term frequency, inverse document frequency and length.
The k1 parameter saturates repeated terms and the b parameter normalises long documents.
BM25 finds exact identifiers such as CVE-2024-3094 that embedding models often miss.
""",
    "chunking": """# Ingestion
## Chunking
Chunking splits documents into passages small enough to embed precisely.
Recursive chunking tries paragraph breaks first, then lines, then sentences, then words.
Structure-aware chunking prepends the heading breadcrumb so each passage keeps its topic.
""",
    "reranking": """# Reranking
## Cross-Encoders
A cross-encoder reads the query and one passage together and outputs a relevance score.
It is far more accurate than a bi-encoder but too slow to run over the whole corpus.
Rerank the top 20 to 50 fused candidates and keep the best 3 to 5 for the prompt.
""",
    "pgvector": """# Storage
## pgvector
pgvector adds a vector column type and HNSW or IVFFlat indexes to PostgreSQL.
Use vector_cosine_ops with the <=> operator for cosine distance.
Raise hnsw.ef_search to trade query latency for higher recall.
""",
}
