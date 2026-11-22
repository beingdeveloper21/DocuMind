from pathlib import Path
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from documind.config import settings
from documind.core.exceptions import IndexingError
from documind.core.models import DocumentChunk
from documind.retrieval.embeddings import BaseEmbeddingProvider, get_embedding_provider


class ChromaVectorStore:
    """
    Enterprise vector database client managing document embeddings in ChromaDB.
    Supports persistent storage, cosine similarity indexing, and batch ingestion.
    """

    COLLECTION_NAME = "documind_research_corpus"

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        in_memory: bool = False,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.in_memory = in_memory
        self.collection_name = collection_name or self.COLLECTION_NAME

        if self.in_memory:
            self.client = chromadb.Client(ChromaSettings(anonymized_telemetry=False, is_persistent=False))
        else:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
            )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embeds and indexes document chunks in ChromaDB."""
        if not chunks:
            return 0

        try:
            texts = [c.content for c in chunks]
            embeddings = self.embedding_provider.embed_texts(texts)
            ids = [c.chunk_id for c in chunks]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "page_number": c.page_number,
                    "section_header": c.section_header,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ]

            # Upsert into ChromaDB in batches of 200
            batch_size = 200
            for i in range(0, len(chunks), batch_size):
                self.collection.upsert(
                    ids=ids[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size],
                    documents=texts[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                )

            return len(chunks)

        except Exception as e:
            raise IndexingError(f"Failed to index chunks in ChromaDB: {str(e)}") from e

    def query_similarity(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[DocumentChunk, float, List[float]]]:
        """
        Executes dense vector similarity search.
        Returns list of tuples: (DocumentChunk, similarity_score, embedding_vector).
        Cosine distance is normalized to [0, 1] cosine similarity.
        """
        try:
            query_embedding = self.embedding_provider.embed_query(query)
            where_filter = None
            if doc_ids:
                if len(doc_ids) == 1:
                    where_filter = {"doc_id": doc_ids[0]}
                else:
                    where_filter = {"doc_id": {"$in": doc_ids}}

            total_count = self.collection.count()
            if total_count == 0:
                return []

            effective_k = min(top_k, total_count)
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": effective_k,
                "include": ["documents", "metadatas", "distances", "embeddings"],
            }
            if where_filter:
                query_params["where"] = where_filter

            results = self.collection.query(**query_params)

            matched: List[Tuple[DocumentChunk, float, List[float]]] = []
            if not results or not results["ids"] or not results["ids"][0]:
                return []

            ids = results["ids"][0]
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            embeddings = results["embeddings"][0]

            for cid, doc_text, meta, dist, emb in zip(ids, docs, metadatas, distances, embeddings):
                # Chroma cosine distance ranges [0, 2]. Cosine similarity = 1 - (dist / 2) or 1 - dist
                sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                chunk = DocumentChunk(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", ""),
                    doc_name=meta.get("doc_name", ""),
                    page_number=meta.get("page_number", 1),
                    section_header=meta.get("section_header", "General"),
                    content=doc_text,
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                )
                matched.append((chunk, sim, emb))

            return matched

        except Exception as e:
            raise IndexingError(f"Vector search failed: {str(e)}") from e

    def delete_by_document(self, doc_id: str) -> None:
        """Deletes all chunks associated with a specific document ID."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            raise IndexingError(f"Failed to delete document {doc_id} from Chroma: {str(e)}") from e

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieves a single chunk by chunk ID."""
        res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if res and res["ids"] and len(res["ids"]) > 0:
            doc_text = res["documents"][0]
            meta = res["metadatas"][0]
            return DocumentChunk(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                doc_name=meta.get("doc_name", ""),
                page_number=meta.get("page_number", 1),
                section_header=meta.get("section_header", "General"),
                content=doc_text,
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            )
        return None

    def get_total_chunks(self) -> int:
        """Returns total number of chunks currently indexed in ChromaDB."""
        return self.collection.count()
