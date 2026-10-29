"""
Unit test suite for the Retrieval-Augmented Generation (RAG) Architecture Engine.
Tests cover Hybrid Search, RRF, Cross-Encoder Reranker, CRAG fallbacks, Self-RAG, and Roadmap integrity.
"""

import unittest
from src.learn_rag.rag import (
    DocumentChunk,
    HybridRetriever,
    CrossEncoderReranker,
    CorrectiveRAG,
    SelfRAG,
    RAGPipeline
)
from src.learn_rag.roadmap import (
    RAG_STAGES,
    get_completed_count,
    get_progress_percentage
)


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        self.chunk1 = DocumentChunk(
            text="Reciprocal Rank Fusion RRF fuses dense and sparse vectors effectively.",
            chunk_id="c1"
        )
        self.chunk2 = DocumentChunk(
            text="Cross-Encoder models apply joint attention to pairs of sequences.",
            chunk_id="c2"
        )
        self.chunk3 = DocumentChunk(
            text="Corrective RAG CRAG checks document relevance before generation.",
            chunk_id="c3"
        )

    def test_document_chunk_term_frequency(self):
        tf = self.chunk1.get_term_freq()
        self.assertIn("reciprocal", tf)
        self.assertIn("rrf", tf)
        self.assertGreater(tf["reciprocal"], 0.0)

    def test_hybrid_retriever_rrf(self):
        retriever = HybridRetriever(rrf_k=60)
        retriever.add_chunks([self.chunk1, self.chunk2, self.chunk3])

        results = retriever.retrieve("Reciprocal Rank Fusion RRF", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertGreater(results[0].score, 0.0)

    def test_cross_encoder_reranker(self):
        reranker = CrossEncoderReranker(cutoff_threshold=0.05)
        candidates = [self.chunk1, self.chunk2]

        reranked = reranker.rerank("joint attention sequences", candidates)
        self.assertTrue(len(reranked) >= 1)
        self.assertEqual(reranked[0].chunk_id, "c2")

    def test_corrective_rag_evaluation_and_fallback(self):
        # High confidence
        chunk_high = DocumentChunk(text="High")
        chunk_high.score = 0.50
        grade, conf = CorrectiveRAG.evaluate([chunk_high])
        self.assertEqual(grade, "CORRECT")

        # Low confidence
        chunk_low = DocumentChunk(text="Low")
        chunk_low.score = 0.01
        grade_low, conf_low = CorrectiveRAG.evaluate([chunk_low])
        self.assertEqual(grade_low, "INCORRECT")

        # Fallback search
        fallback = CorrectiveRAG.fallback_search("test query")
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0].metadata["source"], "external_web_fallback")

    def test_self_rag_reflection_tokens(self):
        chunk = DocumentChunk(text="Relevant context")
        chunk.score = 0.50
        res = SelfRAG.generate_with_reflection("sample query", [chunk])

        tokens = res["reflection_tokens"]
        self.assertEqual(tokens["retrieve"], "[Retrieve: Yes]")
        self.assertEqual(tokens["is_rel"], "[IsRel: Relevant]")
        self.assertIn("[IsUse:", tokens["is_use"])

    def test_rag_pipeline_end_to_end(self):
        retriever = HybridRetriever(rrf_k=60)
        retriever.add_chunks([self.chunk1, self.chunk2, self.chunk3])
        pipeline = RAGPipeline(retriever=retriever)

        out = pipeline.query("What is Reciprocal Rank Fusion?")
        self.assertIn("query", out)
        self.assertIn("crag_grade", out)
        self.assertIn("response", out)

    def test_roadmap_7_stages(self):
        self.assertEqual(len(RAG_STAGES), 7)
        self.assertEqual(get_completed_count(), 7)
        self.assertEqual(get_progress_percentage(), 100)


if __name__ == "__main__":
    unittest.main()
