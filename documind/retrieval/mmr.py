from typing import List, Tuple
import numpy as np
from documind.core.models import DocumentChunk, ScoredChunk


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two float vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    candidate_embeddings: List[np.ndarray],
    candidates: List[ScoredChunk],
    lambda_param: float = 0.7,
    k: int = 5,
) -> List[ScoredChunk]:
    """
    Reranks candidate chunks using Maximal Marginal Relevance (MMR) diversity algorithm.
    Eliminates redundant/duplicate context passages from similar sections.

    Formula:
        MMR = argmax_{d in R \\ S} [ lambda * Sim(d, Query) - (1 - lambda) * max_{s in S} Sim(d, s) ]

    Args:
        query_embedding: Vector representation of the user query.
        candidate_embeddings: List of embedding vectors corresponding to candidates.
        candidates: Candidate ScoredChunk items.
        lambda_param: Diversity factor (1.0 = pure relevance, 0.0 = maximal diversity).
        k: Maximum number of diverse chunks to return.

    Returns:
        List of ScoredChunk marked with mmr_selected=True and their assigned mmr_rank.
    """
    if not candidates or len(candidates) == 0:
        return []

    effective_k = min(k, len(candidates))
    q_emb = np.array(query_embedding, dtype=np.float32)

    # Precompute query similarities for all candidates
    query_similarities = []
    cand_embs = []
    for c_emb in candidate_embeddings:
        c_arr = np.array(c_emb, dtype=np.float32)
        cand_embs.append(c_arr)
        query_similarities.append(cosine_similarity(q_emb, c_arr))

    selected_indices: List[int] = []
    unselected_indices: List[int] = list(range(len(candidates)))

    # Step 1: Pick candidate with highest query similarity
    best_first_idx = int(np.argmax(query_similarities))
    selected_indices.append(best_first_idx)
    unselected_indices.remove(best_first_idx)

    # Step 2: Iteratively pick remaining candidates using MMR formula
    while len(selected_indices) < effective_k and unselected_indices:
        best_mmr_score = -float("inf")
        best_candidate_idx = -1

        for u_idx in unselected_indices:
            sim_to_query = query_similarities[u_idx]

            # Find max similarity to any already selected document
            max_sim_to_selected = max(
                cosine_similarity(cand_embs[u_idx], cand_embs[s_idx])
                for s_idx in selected_indices
            )

            # MMR equation
            mmr_score = (lambda_param * sim_to_query) - ((1.0 - lambda_param) * max_sim_to_selected)

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate_idx = u_idx

        if best_candidate_idx != -1:
            selected_indices.append(best_candidate_idx)
            unselected_indices.remove(best_candidate_idx)
        else:
            break

    # Build final list of ScoredChunks with MMR metadata
    result: List[ScoredChunk] = []
    for rank, sel_idx in enumerate(selected_indices, start=1):
        sc = candidates[sel_idx].model_copy()
        sc.mmr_selected = True
        sc.mmr_rank = rank
        result.append(sc)

    return result
