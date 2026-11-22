from pathlib import Path
import pytest


def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["project"] == "DocuMind"


def test_document_lifecycle_and_query_flow(client):
    sample_pdf = Path("sample_data/neural_retrieval_and_adaptive_rag.pdf")
    assert sample_pdf.exists()

    # 1. Upload Document
    with open(sample_pdf, "rb") as f:
        upload_res = client.post(
            "/api/v1/documents/upload",
            files={"file": ("neural_rag_test.pdf", f, "application/pdf")},
        )
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["doc_id"]
    assert upload_data["filename"] == "neural_rag_test.pdf"
    assert upload_data["total_pages"] == 3
    assert upload_data["total_chunks"] > 0

    # 2. List Documents
    list_res = client.get("/api/v1/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["doc_id"] == doc_id for d in docs)

    # 3. Synchronous Query
    query_res = client.post(
        "/api/v1/query",
        json={"query": "What is the MMR diversity parameter and hallucination reduction?", "top_k": 5, "final_k": 3},
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert len(query_data["answer"]) > 0
    assert len(query_data["retrieved_chunks"]) > 0
    assert query_data["groundedness_score"] > 0.0

    # 4. Retrieval Inspection Endpoint
    inspect_res = client.post(
        "/api/v1/retrieval/inspect",
        json={"query": "hybrid retrieval scoring", "alpha": 0.6, "diversity_lambda": 0.7},
    )
    assert inspect_res.status_code == 200
    inspect_data = inspect_res.json()
    assert inspect_data["alpha"] == 0.6
    assert inspect_data["diversity_lambda"] == 0.7
    assert len(inspect_data["retrieved_chunks"]) > 0

    # 5. Autonomous Research Endpoint
    research_res = client.post(
        "/api/v1/research",
        json={"objective": "Compare neural dense embeddings and BM25 keywords", "sub_query_limit": 2},
    )
    assert research_res.status_code == 200
    research_data = research_res.json()
    assert len(research_data["decomposed_queries"]) == 2
    assert len(research_data["executive_synthesis"]) > 0

    # 6. Streaming Query Endpoint (SSE)
    stream_res = client.post(
        "/api/v1/query/stream",
        json={"query": "Summarize empirical evaluation findings", "top_k": 5, "final_k": 3},
    )
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers["content-type"]
    stream_content = stream_res.text
    assert "event: context" in stream_content
    assert "event: token" in stream_content
    assert "event: citations" in stream_content
    assert "event: done" in stream_content

    # 7. Delete Document
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200

    # Verify deleted
    list_after = client.get("/api/v1/documents").json()
    assert not any(d["doc_id"] == doc_id for d in list_after)
