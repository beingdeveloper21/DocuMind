import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from documind.config import settings
from documind.core.exceptions import IndexingError
from documind.core.models import DocumentChunk


class BM25Index:
    """
    High-performance BM25 lexical keyword search index.
    Tokenizes corpus passages, builds inverted index statistics,
    and supports persistent disk serialization.
    """

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or settings.BM25_INDEX_PATH
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._doc_id_to_indices: Dict[str, List[int]] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Normalize and tokenize text for BM25 term frequency analysis."""
        cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
        return tokens

    def add_chunks(self, new_chunks: List[DocumentChunk]) -> None:
        """Adds new document chunks and rebuilds the BM25 model."""
        if not new_chunks:
            return

        self.chunks.extend(new_chunks)
        new_tokenized = [self.tokenize(c.content) for c in new_chunks]
        self.corpus_tokens.extend(new_tokenized)

        # Re-build BM25 index
        self._rebuild_index()
        self.save()

    def remove_document(self, doc_id: str) -> None:
        """Removes all chunks belonging to a document and rebuilds BM25."""
        retained_chunks = []
        retained_tokens = []

        for chunk, tokens in zip(self.chunks, self.corpus_tokens):
            if chunk.doc_id != doc_id:
                retained_chunks.append(chunk)
                retained_tokens.append(tokens)

        self.chunks = retained_chunks
        self.corpus_tokens = retained_tokens
        self._rebuild_index()
        self.save()

    def _rebuild_index(self) -> None:
        """Internal helper to construct BM25Okapi model from corpus tokens."""
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def query(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Executes BM25 keyword query.
        Returns list of (DocumentChunk, normalized_bm25_score).
        Scores are normalized to [0, 1] range.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        if len(raw_scores) == 0:
            return []

        max_score = float(np.max(raw_scores))
        if max_score <= 0.0:
            return []

        scored_pairs: List[Tuple[DocumentChunk, float]] = []
        for idx, score in enumerate(raw_scores):
            chunk = self.chunks[idx]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue

            if score > 0.0:
                # Min-max normalization against maximum BM25 score in the corpus
                norm_score = float(score / max_score)
                scored_pairs.append((chunk, norm_score))

        # Sort descending by normalized score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def save(self) -> None:
        """Persists BM25 chunks and tokenized corpus to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "corpus_tokens": self.corpus_tokens,
                    },
                    f,
                )
        except Exception as e:
            raise IndexingError(f"Failed to persist BM25 index: {str(e)}") from e

    def load(self) -> bool:
        """Loads BM25 index from disk if present."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data.get("chunks", [])
                self.corpus_tokens = data.get("corpus_tokens", [])
                self._rebuild_index()
            return True
        except Exception:
            return False

    def get_total_chunks(self) -> int:
        return len(self.chunks)
