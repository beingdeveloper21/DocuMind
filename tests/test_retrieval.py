import numpy as np
import pytest
from documind.core.models import DocumentChunk, ScoredChunk
from documind.retrieval.embeddings import LocalDeterministicEmbeddingProvider
from documind.retrieval.mmr import maximal_marginal_relevance
from documind.retrieval.vector_store import ChromaVectorStore


def test_chroma_vector_store(mock_chunks):
    provider = LocalDeterministicEmbeddingProvider(dimension=128)
    store = ChromaVectorStore(embedding_provider=provider, in_memory=True, collection_name="isolated_test_corpus")
    assert store.get_total_chunks() == 0

    added = store.add_chunks(mock_chunks)
    assert added == len(mock_chunks)
    assert store.get_total_chunks() == len(mock_chunks)

    # Query similarity
    results = store.query_similarity("self-attention transduction architecture", top_k=2)
    assert len(results) > 0
    top_chunk, score, emb = results[0]
    assert "attention" in top_chunk.content.lower() or "transformer" in top_chunk.content.lower()
    assert 0.0 <= score <= 1.0
    assert len(emb) == store.embedding_provider.dimension

    # Delete by document
    store.delete_by_document("doc_2")
    assert store.get_total_chunks() == len(mock_chunks) - 1


def test_bm25_indexing_and_scoring(temp_bm25_index, mock_chunks):
    index = temp_bm25_index
    index.add_chunks(mock_chunks)
    assert index.get_total_chunks() == len(mock_chunks)

    # Query for exact keywords
    results = index.query("parametric memory knowledge", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.doc_name == "rag_survey.pdf"
    assert 0.0 <= score <= 1.0

    # Persistence verification
    new_index = temp_bm25_index.__class__(index_path=index.index_path)
    loaded = new_index.load()
    assert loaded is True
    assert new_index.get_total_chunks() == len(mock_chunks)


def test_mmr_diversity_eliminates_duplicates():
    """
    Validates that MMR penalizes and deprioritizes near-duplicate passages
    in favor of diverse document context.
    """
    provider = LocalDeterministicEmbeddingProvider(dimension=64)

    query = "transformer self-attention architecture"
    q_emb = np.array(provider.embed_query(query), dtype=np.float32)

    # Chunk A and Chunk B are nearly identical duplicates
    chunk_a = DocumentChunk(
        chunk_id="dup_a",
        doc_id="doc_1",
        doc_name="paper1.pdf",
        page_number=1,
        section_header="Method",
        content="The Transformer model is entirely based on self-attention mechanisms without recurrence.",
        chunk_index=0,
    )
    chunk_b = DocumentChunk(
        chunk_id="dup_b",
        doc_id="doc_1",
        doc_name="paper1.pdf",
        page_number=1,
        section_header="Method",
        content="The Transformer architecture relies completely on self-attention mechanisms and dispenses with recurrence.",
        chunk_index=1,
    )
    # Chunk C contains different, diverse complementary knowledge
    chunk_c = DocumentChunk(
        chunk_id="diverse_c",
        doc_id="doc_1",
        doc_name="paper1.pdf",
        page_number=5,
        section_header="Training Details",
        content="We trained on the WMT 2014 English-to-German dataset with Adam optimizer and warmup steps.",
        chunk_index=2,
    )

    candidates = [
        ScoredChunk(chunk=chunk_a, dense_score=0.95, bm25_score=0.90, hybrid_score=0.925),
        ScoredChunk(chunk=chunk_b, dense_score=0.94, bm25_score=0.88, hybrid_score=0.910),
        ScoredChunk(chunk=chunk_c, dense_score=0.75, bm25_score=0.60, hybrid_score=0.675),
    ]

    embs = [
        np.array(provider.embed_query(chunk_a.content), dtype=np.float32),
        np.array(provider.embed_query(chunk_b.content), dtype=np.float32),
        np.array(provider.embed_query(chunk_c.content), dtype=np.float32),
    ]

    # Run MMR with high diversity demand (lambda=0.5)
    diverse_results = maximal_marginal_relevance(
        query_embedding=q_emb,
        candidate_embeddings=embs,
        candidates=candidates,
        lambda_param=0.5,
        k=2,
    )

    assert len(diverse_results) == 2
    selected_ids = [sc.chunk.chunk_id for sc in diverse_results]

    # Exactly one duplicate was selected
    duplicates_selected = set(selected_ids).intersection({"dup_a", "dup_b"})
    assert len(duplicates_selected) == 1, f"Expected 1 duplicate, got {duplicates_selected}"

    # MMR should pick the diverse chunk_c over the redundant duplicate!
    assert "diverse_c" in selected_ids, f"MMR failed to select diverse chunk! Selected: {selected_ids}"


def test_hybrid_retriever_pipeline(hybrid_engine):
    results = hybrid_engine.retrieve(
        query="retrieval augmented non-parametric memory",
        top_k=4,
        final_k=2,
        apply_mmr=True,
    )

    assert len(results) > 0
    top = results[0]
    assert top.chunk.doc_name == "rag_survey.pdf"
    assert top.hybrid_score > 0.0
    assert top.mmr_selected is True
    assert top.mmr_rank == 1
