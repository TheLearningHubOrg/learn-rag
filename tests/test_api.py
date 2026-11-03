from fastapi.testclient import TestClient

from app.main import create_app


def client() -> TestClient:
    return TestClient(create_app())


def test_health_reports_seeded_chunks():
    res = client().get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["chunks"] >= 5


def test_search_returns_fused_hits_with_both_ranks():
    res = client().post("/search", json={"query": "reciprocal rank fusion k 60", "top_k": 3})
    assert res.status_code == 200
    top = res.json()[0]
    assert top["chunk_id"].startswith("rrf#")
    assert top["dense_rank"] is not None and top["sparse_rank"] is not None


def test_ingest_then_ask_then_delete():
    c = client()
    doc = {
        "doc_id": "runbook",
        "markdown": "# Runbook\n## Pager\nThe on-call rotation changes every Monday at 09:00 UTC.",
    }
    assert c.post("/documents", json=doc).status_code == 201
    ask = c.post("/ask", json={"query": "When does the on-call rotation change?"}).json()
    assert ask["grade"] == "correct"
    assert ask["citations"] == ["runbook#0"]
    assert c.delete("/documents/runbook").json()["chunks_removed"] == 1
    assert c.delete("/documents/runbook").status_code == 404


def test_validation_rejects_empty_query():
    assert client().post("/search", json={"query": ""}).status_code == 422
