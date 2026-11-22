from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from documind.api import routes
from documind.config import settings
from documind.reasoning.citation_verifier import CitationVerifier
from documind.reasoning.llm import LLMClient
from documind.reasoning.research_agent import AutonomousResearchAgent
from documind.retrieval.bm25_index import BM25Index
from documind.retrieval.hybrid_retriever import HybridRetriever
from documind.retrieval.vector_store import ChromaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and binds singleton database clients and engines upon startup."""
    # Ensure all directories exist
    settings.ensure_directories()

    # Initialize Vector Store & BM25 Index
    routes.vector_store = ChromaVectorStore(persist_directory=settings.CHROMA_PERSIST_DIR)
    routes.bm25_index = BM25Index(index_path=settings.BM25_INDEX_PATH)
    routes.bm25_index.load()

    # Initialize Hybrid Retriever
    routes.hybrid_retriever = HybridRetriever(
        vector_store=routes.vector_store,
        bm25_index=routes.bm25_index,
        alpha=settings.HYBRID_ALPHA,
        diversity_lambda=settings.MMR_DIVERSITY_LAMBDA,
    )

    # Initialize Reasoning & LLM Clients
    routes.llm_client = LLMClient()
    routes.citation_verifier = CitationVerifier()
    routes.research_agent = AutonomousResearchAgent(
        retriever=routes.hybrid_retriever,
        llm_client=routes.llm_client,
        citation_verifier=routes.citation_verifier,
    )

    # Load Document Catalog
    routes.load_catalog()

    yield

    # Clean shutdown tasks
    if routes.bm25_index:
        routes.bm25_index.save()
    routes.save_catalog()


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="DocuMind API",
        description="Autonomous Research Intelligence & Hybrid RAG System for Academic Literature.",
        version=settings.VERSION,
        lifespan=lifespan,
    )

    # Configure CORS for enterprise integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(routes.router)

    # Mount static assets for frontend web UI
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
