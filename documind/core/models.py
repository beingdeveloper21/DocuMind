from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata describing an uploaded PDF document."""
    doc_id: str
    filename: str
    total_pages: int
    file_size_bytes: int
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_chunks: int = 0
    sha256_hash: str = ""


class DocumentChunk(BaseModel):
    """A granular chunk of an academic paper with exact source-page tracking."""
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    section_header: str = "General"
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A retrieved chunk enriched with multi-stage ranking metrics."""
    chunk: DocumentChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    mmr_selected: bool = False
    mmr_rank: Optional[int] = None


class Citation(BaseModel):
    """A verified source-page citation anchor."""
    doc_name: str
    page_number: int
    section_header: str
    snippet: str
    is_verified: bool = True
    grounding_confidence: float = 1.0


class GroundedAnswer(BaseModel):
    """Structured QA response grounded with source citations."""
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    groundedness_score: float = 1.0
    retrieved_chunks: List[ScoredChunk] = Field(default_factory=list)
    latency_ms: float = 0.0


class ResearchSubQuery(BaseModel):
    """A decomposed sub-question generated during autonomous research."""
    sub_query: str
    rationale: str
    retrieved_evidence: List[ScoredChunk] = Field(default_factory=list)
    sub_answer: str = ""


class AutonomousResearchReport(BaseModel):
    """Comprehensive multi-paper comparative intelligence report."""
    objective: str
    decomposed_queries: List[ResearchSubQuery]
    comparative_matrix: Dict[str, Any] = Field(default_factory=dict)
    executive_synthesis: str
    citations: List[Citation] = Field(default_factory=list)
    papers_consulted: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
