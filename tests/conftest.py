import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from documind.api.app import app
from documind.config import settings
from documind.core.models import DocumentChunk
from documind.retrieval.bm25_index import BM25Index
from documind.retrieval.embeddings import LocalDeterministicEmbeddingProvider
from documind.retrieval.hybrid_retriever import HybridRetriever
from documind.retrieval.vector_store import ChromaVectorStore


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("documind_test_data")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            doc_name="attention_paper.pdf",
            page_number=1,
            section_header="Introduction",
            content="Recurrent models typically factor computation along the symbol positions of the input and output sequences.",
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            doc_id="doc_1",
            doc_name="attention_paper.pdf",
            page_number=2,
            section_header="Model Architecture",
            content="The Transformer is the first transduction model relying entirely on self-attention to compute representations.",
            chunk_index=1,
        ),
        # Near duplicate chunk to test MMR diversity elimination
        DocumentChunk(
            chunk_id="chunk_3",
            doc_id="doc_1",
            doc_name="attention_paper.pdf",
            page_number=2,
            section_header="Model Architecture",
            content="The Transformer relies entirely on self-attention mechanisms to compute input and output sequence representations.",
            chunk_index=2,
        ),
        DocumentChunk(
            chunk_id="chunk_4",
            doc_id="doc_2",
            doc_name="rag_survey.pdf",
            page_number=3,
            section_header="Retrieval Augmentation",
            content="Retrieval-Augmented Generation blends parametric memory in large language models with non-parametric external knowledge.",
            chunk_index=3,
        ),
    ]


@pytest.fixture
def in_memory_vector_store():
    provider = LocalDeterministicEmbeddingProvider(dimension=128)
    store = ChromaVectorStore(embedding_provider=provider, in_memory=True)
    return store


@pytest.fixture
def temp_bm25_index(tmp_path):
    index_file = tmp_path / "test_bm25.pkl"
    index = BM25Index(index_path=index_file)
    return index


@pytest.fixture
def hybrid_engine(in_memory_vector_store, temp_bm25_index, mock_chunks):
    in_memory_vector_store.add_chunks(mock_chunks)
    temp_bm25_index.add_chunks(mock_chunks)
    return HybridRetriever(
        vector_store=in_memory_vector_store,
        bm25_index=temp_bm25_index,
        alpha=0.5,
        diversity_lambda=0.7,
    )


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
