import json
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Dict, List
import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from documind.api.schemas import (
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    RetrievalInspectRequest,
    RetrievalInspectResponse,
)
from documind.config import settings
from documind.core.models import (
    AutonomousResearchReport,
    DocumentMetadata,
    GroundedAnswer,
)
from documind.ingestion.chunker import SemanticChunker
from documind.ingestion.parser import PDFDocumentParser
from documind.reasoning.citation_verifier import CitationVerifier
from documind.reasoning.llm import LLMClient
from documind.reasoning.research_agent import AutonomousResearchAgent
from documind.retrieval.bm25_index import BM25Index
from documind.retrieval.hybrid_retriever import HybridRetriever
from documind.retrieval.vector_store import ChromaVectorStore

router = APIRouter(prefix="/api/v1")

# Global singleton component references (initialized in lifespan)
vector_store: ChromaVectorStore = None
bm25_index: BM25Index = None
hybrid_retriever: HybridRetriever = None
llm_client: LLMClient = None
citation_verifier: CitationVerifier = None
research_agent: AutonomousResearchAgent = None
parser: PDFDocumentParser = PDFDocumentParser()
chunker: SemanticChunker = SemanticChunker()

# Persistent registry file for uploaded document metadata
DOCUMENTS_REGISTRY_PATH = settings.DATA_DIR / "documents_registry.json"
documents_catalog: Dict[str, DocumentMetadata] = {}


def load_catalog() -> None:
    """Loads document catalog from disk."""
    global documents_catalog
    if DOCUMENTS_REGISTRY_PATH.exists():
        try:
            with open(DOCUMENTS_REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents_catalog = {
                    doc_id: DocumentMetadata(**meta) for doc_id, meta in data.items()
                }
        except Exception:
            documents_catalog = {}


def save_catalog() -> None:
    """Saves document catalog to disk."""
    try:
        DOCUMENTS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DOCUMENTS_REGISTRY_PATH, "w", encoding="utf-8") as f:
            data = {k: v.model_dump(mode="json") for k, v in documents_catalog.items()}
            json.dump(data, f, indent=2)
    except Exception as e:
        pass


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests an academic research paper in PDF format.
    Parses layout and section boundaries, chunks semantically, and indexes
    into both dense ChromaDB and lexical BM25 stores.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported for ingestion.",
        )

    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    file_path = settings.UPLOAD_DIR / f"{doc_id}_{file.filename}"

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        async with aiofiles.open(file_path, "wb") as out_file:
            await out_file.write(content)

        # Parse PDF
        metadata, parsed_pages = parser.parse(content, filename=file.filename, doc_id=doc_id)

        # Semantically chunk pages
        chunks = chunker.chunk_pages(doc_id=doc_id, doc_name=file.filename, pages=parsed_pages)
        metadata.total_chunks = len(chunks)

        # Index in ChromaDB
        vector_store.add_chunks(chunks)

        # Index in BM25
        bm25_index.add_chunks(chunks)

        # Update and save catalog
        documents_catalog[doc_id] = metadata
        save_catalog()

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=metadata.filename,
            total_pages=metadata.total_pages,
            total_chunks=metadata.total_chunks,
            file_size_bytes=metadata.file_size_bytes,
            sha256_hash=metadata.sha256_hash,
        )

    except HTTPException:
        raise
    except Exception as e:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


@router.post("/sample/load", response_model=List[DocumentUploadResponse], status_code=status.HTTP_201_CREATED)
async def load_sample_papers():
    """
    Ingests bundled sample academic papers for immediate demonstration and testing.
    """
    sample_dir = Path(__file__).resolve().parent.parent.parent / "sample_data"
    sample_files = list(sample_dir.glob("*.pdf"))

    if not sample_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No sample PDF files found in sample_data directory.",
        )

    responses: List[DocumentUploadResponse] = []

    for pdf_path in sample_files:
        # Check if already indexed
        already_loaded = any(d.filename == pdf_path.name for d in documents_catalog.values())
        if already_loaded:
            continue

        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        content = pdf_path.read_bytes()

        # Copy to uploads
        dest_path = settings.UPLOAD_DIR / f"{doc_id}_{pdf_path.name}"
        dest_path.write_bytes(content)

        # Parse & Chunk
        metadata, parsed_pages = parser.parse(content, filename=pdf_path.name, doc_id=doc_id)
        chunks = chunker.chunk_pages(doc_id=doc_id, doc_name=pdf_path.name, pages=parsed_pages)
        metadata.total_chunks = len(chunks)

        # Index in ChromaDB & BM25
        vector_store.add_chunks(chunks)
        bm25_index.add_chunks(chunks)

        documents_catalog[doc_id] = metadata
        responses.append(
            DocumentUploadResponse(
                doc_id=doc_id,
                filename=metadata.filename,
                total_pages=metadata.total_pages,
                total_chunks=metadata.total_chunks,
                file_size_bytes=metadata.file_size_bytes,
                sha256_hash=metadata.sha256_hash,
            )
        )

    save_catalog()
    return responses


@router.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """Lists all currently indexed research papers with metadata."""
    return list(documents_catalog.values())


@router.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(doc_id: str):
    """Deletes an indexed research paper from ChromaDB and BM25 index."""
    if doc_id not in documents_catalog:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {doc_id} not found.",
        )

    try:
        # Delete from ChromaDB
        vector_store.delete_by_document(doc_id)

        # Delete from BM25
        bm25_index.remove_document(doc_id)

        # Remove file from disk
        meta = documents_catalog.pop(doc_id)
        save_catalog()
        file_path = settings.UPLOAD_DIR / f"{doc_id}_{meta.filename}"
        if file_path.exists():
            file_path.unlink(missing_ok=True)

        return {"message": f"Document '{meta.filename}' (ID: {doc_id}) successfully removed."}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )


@router.post("/query", response_model=QueryResponse)
async def query_academic_corpus(payload: QueryRequest):
    """
    Standard synchronous conversational QA endpoint.
    Retrieves context using hybrid search with MMR diversity,
    synthesizes a grounded answer, and validates citations.
    """
    start_time = time.perf_counter()

    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    # Hybrid retrieval with MMR diversity re-ranking
    retriever_instance = hybrid_retriever
    if payload.alpha is not None or payload.diversity_lambda is not None:
        retriever_instance = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            alpha=payload.alpha,
            diversity_lambda=payload.diversity_lambda,
        )

    retrieved = retriever_instance.retrieve(
        query=payload.query,
        top_k=payload.top_k,
        final_k=payload.final_k,
        doc_ids=payload.doc_ids,
        apply_mmr=True,
    )

    # Generate answer
    raw_answer = await llm_client.generate_answer(
        query=payload.query,
        scored_chunks=retrieved,
        chat_history=payload.chat_history,
    )

    # Verify citations
    citations, groundedness = citation_verifier.verify_and_extract(raw_answer, retrieved)
    latency = round((time.perf_counter() - start_time) * 1000.0, 2)

    return QueryResponse(
        query=payload.query,
        answer=raw_answer,
        citations=citations,
        groundedness_score=groundedness,
        retrieved_chunks=retrieved,
        latency_ms=latency,
    )


@router.post("/query/stream")
async def stream_academic_query(payload: QueryRequest):
    """
    Asynchronous streaming endpoint using Server-Sent Events (SSE).
    Streams answer tokens in real time, followed by verified citation metadata.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()

        # Hybrid retrieval with MMR
        retrieved = hybrid_retriever.retrieve(
            query=payload.query,
            top_k=payload.top_k,
            final_k=payload.final_k,
            doc_ids=payload.doc_ids,
            apply_mmr=True,
        )

        # Emit retrieved context event
        context_data = [
            {
                "chunk_id": sc.chunk.chunk_id,
                "doc_name": sc.chunk.doc_name,
                "page": sc.chunk.page_number,
                "section": sc.chunk.section_header,
                "hybrid_score": sc.hybrid_score,
                "dense_score": sc.dense_score,
                "bm25_score": sc.bm25_score,
                "mmr_rank": sc.mmr_rank,
                "content": sc.chunk.content[:250] + "...",
            }
            for sc in retrieved
        ]
        yield f"event: context\ndata: {json.dumps(context_data)}\n\n"

        # Stream answer tokens
        full_answer_accumulator = []
        async for token in llm_client.stream_answer(
            query=payload.query,
            scored_chunks=retrieved,
            chat_history=payload.chat_history,
        ):
            full_answer_accumulator.append(token)
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        full_answer = "".join(full_answer_accumulator)

        # Verify citations
        citations, groundedness = citation_verifier.verify_and_extract(full_answer, retrieved)
        latency = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Emit citation and completion events
        citations_data = [c.model_dump(mode="json") for c in citations]
        yield f"event: citations\ndata: {json.dumps({'citations': citations_data, 'groundedness_score': groundedness})}\n\n"
        yield f"event: done\ndata: {json.dumps({'latency_ms': latency, 'total_tokens': len(full_answer.split())})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/research", response_model=AutonomousResearchReport)
async def autonomous_research(payload: ResearchRequest):
    """
    Autonomous Research Intelligence Mode.
    Decomposes multi-faceted research hypotheses into sub-queries,
    orchestrates hybrid retrieval, synthesizes a comparative report,
    and cross-verifies all citations.
    """
    if not payload.objective.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Research objective cannot be empty.",
        )

    report = await research_agent.execute_research(
        objective=payload.objective,
        doc_ids=payload.doc_ids,
        sub_query_limit=payload.sub_query_limit,
    )
    return report


@router.post("/retrieval/inspect", response_model=RetrievalInspectResponse)
async def inspect_retrieval(payload: RetrievalInspectRequest):
    """
    Inspects multi-stage hybrid retrieval mechanics.
    Exposes raw Dense similarity, BM25 keyword score, blended hybrid score,
    and MMR diversity rank assignments.
    """
    custom_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        alpha=payload.alpha,
        diversity_lambda=payload.diversity_lambda,
    )

    retrieved = custom_retriever.retrieve(
        query=payload.query,
        top_k=payload.top_k,
        final_k=payload.final_k,
        doc_ids=payload.doc_ids,
        apply_mmr=True,
    )

    return RetrievalInspectResponse(
        query=payload.query,
        alpha=payload.alpha,
        diversity_lambda=payload.diversity_lambda,
        total_candidates=len(retrieved),
        retrieved_chunks=retrieved,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and database statistics."""
    chroma_count = vector_store.get_total_chunks() if vector_store else 0
    bm25_count = bm25_index.get_total_chunks() if bm25_index else 0

    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        total_documents=len(documents_catalog),
        total_chunks_chroma=chroma_count,
        total_chunks_bm25=bm25_count,
        openai_configured=bool(llm_client and llm_client.is_live),
    )
