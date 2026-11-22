from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from documind.core.models import (
    AutonomousResearchReport,
    Citation,
    DocumentMetadata,
    GroundedAnswer,
    ScoredChunk,
)


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful PDF ingestion."""
    doc_id: str
    filename: str
    total_pages: int
    total_chunks: int
    file_size_bytes: int
    sha256_hash: str
    message: str = "Document successfully parsed, chunked, and indexed in ChromaDB and BM25."


class QueryRequest(BaseModel):
    """Payload for conversational research QA."""
    query: str = Field(..., description="The academic research question to answer.")
    doc_ids: Optional[List[str]] = Field(None, description="Optional list of document IDs to restrict search to.")
    top_k: int = Field(10, description="Number of candidate chunks to retrieve from hybrid search.")
    final_k: int = Field(5, description="Number of diverse chunks to select after MMR re-ranking.")
    alpha: Optional[float] = Field(None, description="Dense vs Lexical weight (0.0 to 1.0). None uses server default.")
    diversity_lambda: Optional[float] = Field(None, description="MMR diversity trade-off (0.0 to 1.0).")
    chat_history: str = Field("", description="Optional conversation context from previous turns.")


class QueryResponse(GroundedAnswer):
    """Complete grounded answer with verified source citations."""
    pass


class ResearchRequest(BaseModel):
    """Payload for autonomous multi-step deep research intelligence."""
    objective: str = Field(..., description="High-level research objective or comparative hypothesis.")
    doc_ids: Optional[List[str]] = Field(None, description="Optional subset of document IDs to investigate.")
    sub_query_limit: int = Field(3, description="Maximum number of decomposed research sub-questions.")


class RetrievalInspectRequest(BaseModel):
    """Payload to inspect hybrid retrieval and MMR diversity metrics."""
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: int = 10
    final_k: int = 5
    alpha: float = 0.5
    diversity_lambda: float = 0.7


class RetrievalInspectResponse(BaseModel):
    """Detailed score inspection for BM25, Dense similarity, and MMR selection."""
    query: str
    alpha: float
    diversity_lambda: float
    total_candidates: int
    retrieved_chunks: List[ScoredChunk]


class HealthResponse(BaseModel):
    """System health and corpus status."""
    status: str
    project: str
    version: str
    total_documents: int
    total_chunks_chroma: int
    total_chunks_bm25: int
    openai_configured: bool
