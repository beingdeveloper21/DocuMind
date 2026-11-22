# DocuMind: Autonomous Research Intelligence & Hybrid RAG

DocuMind is an enterprise-grade Retrieval-Augmented Generation (RAG) platform designed for technical research literature and PDF analysis.

## Key Features

1. **PDF Ingestion Engine**
   - Multi-page layout parsing using PyMuPDF.
   - Section header detection and page-accurate citation tracking.
   - Semantic chunking with configurable overlap.

2. **Hybrid Retrieval System**
   - ChromaDB dense vector indexing with cosine distance.
   - BM25 lexical keyword search with disk persistence.
   - Blended hybrid scoring combining dense and sparse ranks.

3. **Maximal Marginal Relevance (MMR) Re-Ranking**
   - Mathematically balances query similarity and context diversity.
   - Eliminates redundant passages from duplicate document sections.

4. **Grounded Synthesis & Citation Verification**
   - Strict source-page citation tracking: [Doc: <name>, Page: <page>, Section: <section>].
   - Automated groundedness scoring to audit generated statements against source passages.

5. **Asynchronous Streaming FastAPI Server**
   - Server-Sent Events (SSE) streaming verified answer tokens.
   - Real-time emission of citation metadata and latency metrics.

6. **Autonomous Research Intelligence Agent**
   - Automated query decomposition into sub-investigations.
   - Multi-paper comparative synthesis and executive reporting.

7. **Interactive Web Application**
   - Single-page responsive interface.
   - Live Hybrid Search Inspector showing ChromaDB vs BM25 weights and MMR rank adjustments.
   - Interactive evidence modals displaying verbatim source passages.

## Quickstart

### Prerequisites
- Python 3.10+
- (Optional) Docker

### 1. Install Dependencies
`ash
pip install -r requirements.txt
`

### 2. Configure Environment (Optional)
Copy .env.example to .env:
`ash
cp .env.example .env
`
*Note: An OpenAI API key can be set in .env. If not provided, DocuMind operates with a built-in local vectorizer and offline synthesis engine for development.*

### 3. Run the Server
`ash
uvicorn documind.api.app:app --host 0.0.0.0 --port 8000 --reload
`
Navigate to http://localhost:8000 to interact with the web interface.

### 4. Run with Docker
`ash
docker compose up --build
`

### 5. Run Automated Tests
`ash
pytest -v
`

## API Reference

- POST /api/v1/documents/upload: Upload and index research PDFs.
- GET /api/v1/documents: List indexed papers and chunk statistics.
- DELETE /api/v1/documents/{doc_id}: Delete a paper and remove indexed chunks.
- POST /api/v1/query: Standard conversational QA with citations.
- POST /api/v1/query/stream: Real-time SSE token and citation streaming.
- POST /api/v1/research: Autonomous multi-step deep research mode.
- POST /api/v1/retrieval/inspect: Inspect dense, BM25, and MMR ranking breakdown.
- POST /api/v1/sample/load: One-click ingestion of sample academic papers.
- GET /api/v1/health: System health and corpus metrics.

## License
MIT
