"""
Hybrid retrieval engine combining BM25, ChromaDB dense embeddings, and MMR diversity.
"""

from documind.retrieval.embeddings import (
    BaseEmbeddingProvider,
    LocalDeterministicEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from documind.retrieval.vector_store import ChromaVectorStore
from documind.retrieval.bm25_index import BM25Index
from documind.retrieval.mmr import maximal_marginal_relevance
from documind.retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "BaseEmbeddingProvider",
    "LocalDeterministicEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
    "ChromaVectorStore",
    "BM25Index",
    "maximal_marginal_relevance",
    "HybridRetriever",
]
