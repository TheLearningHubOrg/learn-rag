from app.acl import allowed_for
from app.agent import multi_hop, render_facts
from app.cache import SemanticCache
from app.context import chunk_cost, pack_context
from app.corpus import SEED_DOCUMENTS
from app.embeddings import HashingEmbedder
from app.pipeline import RagPipeline
from app.rerank import OverlapReranker, rerank


def seeded() -> RagPipeline:
    pipe = RagPipeline()
    for doc_id, text in SEED_DOCUMENTS.items():
        pipe.ingest(doc_id, text)
    return pipe


# ---- Stage 4: reranking ---------------------------------------------------
def test_overlap_reranker_rewards_word_order():
    scores = OverlapReranker().score("rank fusion", ["fusion of rank lists", "rank fusion combines lists"])
    assert scores[1] > scores[0]


def test_rerank_promotes_best_passage_and_applies_threshold():
    pipe = seeded()
    query = "cross-encoder reads the query and one passage together"
    hits = pipe.search(query, top_k=5)
    ranked = rerank(query, hits, OverlapReranker(), top_k=2, min_score=0.5)
    assert ranked[0].chunk.chunk_id == "reranking#0"
    assert all(h.score >= 0.5 for h in ranked)
    assert len(ranked) <= 2


# ---- Stage 5: context packing ---------------------------------------------
def test_pack_context_respects_budget_and_skips_oversized_hits():
    pipe = seeded()
    hits = pipe.search("BM25 k1 parameter", top_k=5)
    budget = chunk_cost(hits[0]) + chunk_cost(hits[1])
    packed = pack_context(hits, budget)
    assert packed[0].chunk.chunk_id == hits[0].chunk.chunk_id
    assert sum(chunk_cost(h) for h in packed) <= budget
    assert pack_context(hits, 0) == []


def test_pack_context_caps_chunks_per_document():
    pipe = RagPipeline(chunk_chars=120)
    pipe.ingest("long", "# Long\n" + " ".join(f"Sentence {i} about vectors." for i in range(40)))
    pipe.ingest("other", "# Other\nVectors are also stored in pgvector.")
    hits = pipe.search("vectors", top_k=10)
    packed = pack_context(hits, 10_000, max_per_doc=2)
    assert sum(1 for h in packed if h.chunk.doc_id == "long") == 2
    assert any(h.chunk.doc_id == "other" for h in packed)


# ---- M3: access control -----------------------------------------------------
def test_acl_filter_hides_documents_from_other_groups_in_both_branches():
    pipe = RagPipeline()
    pipe.ingest("salaries", "# HR\nThe salary band for level L5 is confidential.", {"acl": "hr"})
    pipe.ingest("handbook", "# Handbook\nLevels run from L1 to L7.", {"acl": "everyone, hr"})
    pipe.ingest("unlabelled", "# Draft\nL5 salary draft without labels.")
    as_engineer = pipe.search("L5 salary band", top_k=5, where=allowed_for(["everyone", "eng"]))
    assert [h.chunk.doc_id for h in as_engineer] == ["handbook"]
    as_hr = pipe.search("L5 salary band", top_k=5, where=allowed_for(["hr"]))
    assert {h.chunk.doc_id for h in as_hr} == {"salaries", "handbook"}


# ---- M4: semantic cache ---------------------------------------------------
class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_semantic_cache_hits_on_reworded_query_and_misses_on_new_topic():
    cache = SemanticCache(HashingEmbedder(), threshold=0.9)
    cache.put("What does the BM25 k1 parameter do?", "It saturates term frequency.", "v1")
    assert cache.get("what does the k1 parameter in BM25 do", "v1") == "It saturates term frequency."
    assert cache.get("How do I create an HNSW index?", "v1") is None


def test_semantic_cache_invalidates_on_corpus_version_and_ttl():
    clock = FakeClock()
    cache = SemanticCache(HashingEmbedder(), ttl_seconds=60, clock=clock)
    cache.put("reciprocal rank fusion k", "k is usually 60.", "v1")
    assert cache.get("reciprocal rank fusion k", "v2") is None
    clock.now = 61
    assert cache.get("reciprocal rank fusion k", "v1") is None
    assert len(cache) == 0


def test_semantic_cache_evicts_least_recently_used():
    cache = SemanticCache(HashingEmbedder(), max_entries=2)
    cache.put("alpha question", "a", "v1")
    cache.put("beta question", "b", "v1")
    assert cache.get("alpha question", "v1") == "a"
    cache.put("gamma question", "c", "v1")
    assert cache.get("beta question", "v1") is None
    assert cache.get("alpha question", "v1") == "a"


# ---- Stage 6: multi-hop ---------------------------------------------------
def test_multi_hop_collects_one_cited_fact_per_sub_question():
    trace = multi_hop(
        seeded(),
        [
            "What constant k does reciprocal rank fusion use?",
            "Which BM25 parameter saturates repeated terms?",
        ],
    )
    assert trace.stopped == "done"
    assert [f.chunk_id.split("#")[0] for f in trace.facts] == ["rrf", "bm25"]
    assert "[rrf#0]" in render_facts(trace)


def test_multi_hop_stops_on_budget_and_on_missing_evidence():
    pipe = seeded()
    budget = multi_hop(pipe, ["reciprocal rank fusion", "BM25 k1"], max_searches=1)
    assert budget.stopped == "budget" and len(budget.facts) == 1
    missing = multi_hop(pipe, ["quarterly revenue forecast for Mars colonies", "BM25 k1"])
    assert missing.stopped == "missing" and missing.facts == []
