from typing import Dict, List, Optional
import numpy as np

from documind.config import settings
from documind.core.exceptions import RetrievalError
from documind.core.models import DocumentChunk, ScoredChunk
from documind.retrieval.bm25_index import BM25Index
from documind.retrieval.mmr import maximal_marginal_relevance
from documind.retrieval.vector_store import ChromaVectorStore


class HybridRetriever:
    """
    Enterprise Hybrid Retrieval Engine.
    Combines BM25 keyword search and dense ChromaDB embeddings with
    MMR diversity re-ranking to eliminate duplicate document context.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index,
        alpha: Optional[float] = None,
        diversity_lambda: Optional[float] = None,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        self.diversity_lambda = diversity_lambda if diversity_lambda is not None else settings.MMR_DIVERSITY_LAMBDA

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        final_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        apply_mmr: bool = True,
    ) -> List[ScoredChunk]:
        """
        Executes hybrid retrieval combining dense vector search and sparse BM25,
        followed by MMR diversity re-ranking.
        """
        if not query.strip():
            return []

        try:
            # Step 1: Dense ChromaDB retrieval
            dense_results = self.vector_store.query_similarity(
                query=query,
                top_k=top_k * 2,
                doc_ids=doc_ids,
            )

            # Step 2: Lexical BM25 retrieval
            bm25_results = self.bm25_index.query(
                query=query,
                top_k=top_k * 2,
                doc_ids=doc_ids,
            )

            # Merge and normalize scores
            candidates_map: Dict[str, ScoredChunk] = {}
            candidate_embeddings_map: Dict[str, List[float]] = {}

            # Process dense results
            for chunk, sim_score, emb in dense_results:
                candidates_map[chunk.chunk_id] = ScoredChunk(
                    chunk=chunk,
                    dense_score=float(sim_score),
                    bm25_score=0.0,
                    hybrid_score=0.0,
                )
                candidate_embeddings_map[chunk.chunk_id] = emb

            # Process BM25 results
            for chunk, bm25_score in bm25_results:
                if chunk.chunk_id in candidates_map:
                    candidates_map[chunk.chunk_id].bm25_score = float(bm25_score)
                else:
                    candidates_map[chunk.chunk_id] = ScoredChunk(
                        chunk=chunk,
                        dense_score=0.0,
                        bm25_score=float(bm25_score),
                        hybrid_score=0.0,
                    )
                    # Embed chunk if not already present
                    emb = self.vector_store.embedding_provider.embed_query(chunk.content)
                    candidate_embeddings_map[chunk.chunk_id] = emb

            if not candidates_map:
                return []

            # Step 3: Compute Hybrid Blended Score
            all_candidates: List[ScoredChunk] = []
            for cid, sc in candidates_map.items():
                # Blended Convex Combination: alpha * dense + (1 - alpha) * bm25
                hybrid_score = (self.alpha * sc.dense_score) + ((1.0 - self.alpha) * sc.bm25_score)
                sc.hybrid_score = round(float(hybrid_score), 4)
                all_candidates.append(sc)

            # Sort candidate pool descending by hybrid score
            all_candidates.sort(key=lambda x: x.hybrid_score, reverse=True)
            candidate_pool = all_candidates[:top_k]

            # Step 4: Apply MMR Diversity Re-ranking to eliminate duplicates
            if apply_mmr and len(candidate_pool) > 1:
                query_emb = np.array(self.vector_store.embedding_provider.embed_query(query), dtype=np.float32)
                cand_embs = [
                    np.array(candidate_embeddings_map[sc.chunk.chunk_id], dtype=np.float32)
                    for sc in candidate_pool
                ]
                final_results = maximal_marginal_relevance(
                    query_embedding=query_emb,
                    candidate_embeddings=cand_embs,
                    candidates=candidate_pool,
                    lambda_param=self.diversity_lambda,
                    k=final_k,
                )
                return final_results

            # Fallback if MMR disabled or single candidate
            for idx, sc in enumerate(candidate_pool[:final_k], start=1):
                sc.mmr_selected = True
                sc.mmr_rank = idx
            return candidate_pool[:final_k]

        except Exception as e:
            raise RetrievalError(f"Hybrid retrieval failed: {str(e)}") from e
