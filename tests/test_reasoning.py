import pytest
from documind.core.models import DocumentChunk, ScoredChunk
from documind.reasoning.citation_verifier import CitationVerifier
from documind.reasoning.llm import LLMClient
from documind.reasoning.research_agent import AutonomousResearchAgent


def test_citation_verifier_with_valid_citations():
    verifier = CitationVerifier()

    chunk1 = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_name="neural_paper.pdf",
        page_number=2,
        section_header="Methodology",
        content="We evaluate a hybrid retrieval scoring algorithm with MMR lambda 0.7.",
        chunk_index=0,
    )
    retrieved = [ScoredChunk(chunk=chunk1, dense_score=0.9, bm25_score=0.8, hybrid_score=0.85)]

    answer = "The system uses a diversity parameter of 0.7 [Doc: neural_paper.pdf, Page: 2, Section: Methodology]."

    citations, score = verifier.verify_and_extract(answer, retrieved)

    assert len(citations) == 1
    c = citations[0]
    assert c.doc_name == "neural_paper.pdf"
    assert c.page_number == 2
    assert c.section_header == "Methodology"
    assert c.is_verified is True
    assert score == 1.0


def test_citation_verifier_with_hallucinated_page():
    verifier = CitationVerifier()

    chunk1 = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_name="neural_paper.pdf",
        page_number=2,
        section_header="Methodology",
        content="Valid context from page 2.",
        chunk_index=0,
    )
    retrieved = [ScoredChunk(chunk=chunk1, dense_score=0.9, bm25_score=0.8, hybrid_score=0.85)]

    # Model hallucinated Page 99
    answer = "Unsupported assertion [Doc: neural_paper.pdf, Page: 99, Section: Unknown]."

    citations, score = verifier.verify_and_extract(answer, retrieved)

    assert len(citations) == 1
    assert citations[0].is_verified is False
    assert score == 0.0


@pytest.mark.asyncio
async def test_offline_llm_synthesizer_produces_citations():
    llm = LLMClient()
    chunk1 = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_name="scalability.pdf",
        page_number=3,
        section_header="Benchmarks",
        content="ChromaDB persistent mode achieves under 12ms query latencies for 100k vectors.",
        chunk_index=0,
    )
    retrieved = [ScoredChunk(chunk=chunk1, dense_score=0.9, bm25_score=0.8, hybrid_score=0.85)]

    answer = await llm.generate_answer(query="What is ChromaDB latency?", scored_chunks=retrieved)

    assert "scalability.pdf" in answer
    assert "Page: 3" in answer or "Page 3" in answer


@pytest.mark.asyncio
async def test_autonomous_research_agent(hybrid_engine):
    agent = AutonomousResearchAgent(retriever=hybrid_engine)
    report = await agent.execute_research(
        objective="Analyze self-attention mechanisms and retrieval augmentation",
        sub_query_limit=2,
    )

    assert len(report.decomposed_queries) == 2
    assert len(report.comparative_matrix) == 2
    assert len(report.executive_synthesis) > 0
    assert report.latency_ms > 0.0
