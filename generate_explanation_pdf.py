import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count in footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "DocuMind — Complete Project Guide & Technical Interview Dossier")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)

        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — Prepared for Technical Pair-Programming & System Design Interviews")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


def build_explanation_pdf(output_path="explanation.pdf"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    c_primary = colors.HexColor("#0F172A")    # Slate 900
    c_accent = colors.HexColor("#0284C7")     # Sky 600
    c_teal = colors.HexColor("#0D9488")       # Teal 600
    c_dark = colors.HexColor("#1E293B")       # Slate 800
    c_muted = colors.HexColor("#475569")      # Slate 600
    c_card_bg = colors.HexColor("#F8FAFC")    # Slate 50
    c_border = colors.HexColor("#E2E8F0")     # Slate 200

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=c_primary,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        textColor=c_accent,
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_accent,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=c_dark,
        spaceAfter=6,
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    qa_q_style = ParagraphStyle(
        "QA_Question",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )

    qa_a_style = ParagraphStyle(
        "QA_Answer",
        parent=body_style,
        fontSize=9,
        leading=13.5,
        textColor=c_muted,
        spaceAfter=6,
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0369A1"),
        backColor=colors.HexColor("#F1F5F9"),
        borderPadding=4,
        spaceAfter=6,
    )

    story = []

    # Title Banner
    story.append(Paragraph("DocuMind &mdash; Technical Dossier", title_style))
    story.append(Paragraph("Complete Project Architecture, Working Flow, Tech Stack & Comprehensive Interview Preparation Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=0, spaceAfter=12))

    # SECTION 1: EXECUTIVE SUMMARY
    story.append(Paragraph("1. Executive Summary & Problem Solved", h1_style))
    story.append(Paragraph(
        "<b>What is DocuMind?</b> DocuMind is a production-grade, enterprise <b>Retrieval-Augmented Generation (RAG)</b> "
        "and autonomous research intelligence system specifically architected to analyze complex, unstructured scientific and academic literature in PDF format.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Core Problem Solved:</b> Standard naive RAG systems fail when applied to complex academic papers because:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Multi-Column Layout Mismatch:</b> Naive PDF text extractors read horizontally across columns, interleaving sentences from completely different paragraphs.", bullet_style))
    story.append(Paragraph("&bull; <b>Semantic Drift on Specialized Vocabulary:</b> Dense embedding vector search alone often misses exact model acronyms, mathematical formulas, and technical terminology.", bullet_style))
    story.append(Paragraph("&bull; <b>Context Redundancy & Duplicate Chunks:</b> Top-K vector search frequently retrieves 4 or 5 near-identical paragraphs from adjacent pages, wasting prompt context window and blinding the LLM to complementary information.", bullet_style))
    story.append(Paragraph("&bull; <b>Hallucinated Citations:</b> Standard generative QA models produce plausible-sounding answers with fictitious page references that cannot be audited.", bullet_style))
    story.append(Paragraph(
        "DocuMind resolves all four challenges through <b>PyMuPDF block layout parsing</b>, <b>dual-engine Hybrid Retrieval (Dense ChromaDB + Sparse BM25)</b>, "
        "<b>Maximal Marginal Relevance (MMR) diversity re-ranking</b>, and <b>source-page citation verification</b>.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # SECTION 2: END-TO-END WORKING FLOW
    story.append(Paragraph("2. End-to-End Working Flow (In Simple Plain Language)", h1_style))
    story.append(Paragraph(
        "To explain DocuMind in an interview, walk through this chronological 7-step pipeline:",
        body_style
    ))

    flow_steps = [
        ("Step 1: Ingestion & Layout Parsing",
         "When an academic PDF is uploaded, PyMuPDF (fitz) analyzes bounding boxes to extract text block-by-block. "
         "This preserves true column flow, isolates section headers (e.g. Introduction, Methodology), and records exact page numbers."),
        ("Step 2: Semantic Chunking with Provenance",
         "The raw text is segmented into recursive semantic chunks of 1200 characters with 200-character overlap. "
         "Critically, every single chunk carries immutable provenance metadata: doc_id, filename, page_number, and section_header."),
        ("Step 3: Dual-Engine Indexing (Dense + Lexical)",
         "Each chunk is simultaneously indexed into two databases: (1) ChromaDB for dense vector embeddings using cosine similarity HNSW graphs, "
         "and (2) BM25Okapi for term frequency and inverse document frequency keyword matching."),
        ("Step 4: Query Reception & Hybrid Score Fusion",
         "When a user asks a question, DocuMind queries both ChromaDB and BM25. Scores are normalized to [0, 1] and blended via a convex combination: "
         "Score_hybrid = alpha * Score_dense + (1 - alpha) * Score_bm25. This ensures both conceptual meaning and exact terms are captured."),
        ("Step 5: MMR Diversity Re-Ranking",
         "The candidate pool is re-ranked using Maximal Marginal Relevance (MMR with lambda=0.7). If two candidate chunks are near-duplicates, "
         "MMR penalizes the second one and selects a diverse chunk from a different page or section instead."),
        ("Step 6: Grounded Prompt Synthesis & LLM Reasoning",
         "The diverse context passages are formatted with explicit document headers and injected into a strictly grounded system prompt. "
         "The LLM is required to substantiate assertions with [Doc: filename, Page: p, Section: s] citation anchors."),
        ("Step 7: Verification & Asynchronous SSE Streaming",
         "FastAPI streams answer tokens to the browser in real time via Server-Sent Events (SSE). An automated verifier audits all citations against "
         "indexed chunks, computing a Groundedness Score (0-100%) and updating the live Hybrid Search Inspector UI."),
    ]

    for title, desc in flow_steps:
        story.append(Paragraph(f"<b>{title}:</b> {desc}", bullet_style))
    story.append(Spacer(1, 6))

    # SECTION 3: ARCHITECTURE TABLE
    story.append(Paragraph("3. System Architecture & Component Mapping", h1_style))

    table_data = [
        [Paragraph("<b>Component</b>", body_bold), Paragraph("<b>Module / Implementation</b>", body_bold), Paragraph("<b>Core Responsibility</b>", body_bold)],
        [Paragraph("PDF Layout Parser", body_style), Paragraph("documind.ingestion.parser<br/>PyMuPDF (fitz)", code_style), Paragraph("Block extraction, section header detection, page tracking.", body_style)],
        [Paragraph("Semantic Chunker", body_style), Paragraph("documind.ingestion.chunker<br/>SemanticChunker", code_style), Paragraph("Recursive token splitting, provenance metadata injection.", body_style)],
        [Paragraph("Vector Database", body_style), Paragraph("documind.retrieval.vector_store<br/>ChromaVectorStore (ChromaDB)", code_style), Paragraph("Persistent HNSW indexing with cosine similarity distance.", body_style)],
        [Paragraph("Keyword Index", body_style), Paragraph("documind.retrieval.bm25_index<br/>BM25Index (rank-bm25)", code_style), Paragraph("Inverted index, term-frequency analysis, disk persistence.", body_style)],
        [Paragraph("Hybrid Retriever", body_style), Paragraph("documind.retrieval.hybrid_retriever<br/>HybridRetriever", code_style), Paragraph("Convex score fusion (alpha blending) between dense & BM25.", body_style)],
        [Paragraph("Diversity Engine", body_style), Paragraph("documind.retrieval.mmr<br/>maximal_marginal_relevance", code_style), Paragraph("MMR diversity algorithm eliminating duplicate passages.", body_style)],
        [Paragraph("Attribution Verifier", body_style), Paragraph("documind.reasoning.citation_verifier<br/>CitationVerifier", code_style), Paragraph("Regex citation parsing, passage cross-validation, score audit.", body_style)],
        [Paragraph("Research Agent", body_style), Paragraph("documind.reasoning.research_agent<br/>AutonomousResearchAgent", code_style), Paragraph("Query decomposition, parallel search, comparative matrix.", body_style)],
        [Paragraph("API & Streaming", body_style), Paragraph("documind.api.app, routes<br/>FastAPI + SSE Streaming", code_style), Paragraph("Asynchronous REST endpoints, Server-Sent Events token stream.", body_style)],
        [Paragraph("User Interface", body_style), Paragraph("documind.static<br/>HTML5, CSS3, Vanilla JS", code_style), Paragraph("Live inspector, citation popovers, drag-and-drop PDF upload.", body_style)],
    ]

    t = Table(table_data, colWidths=[110, 150, 244])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_card_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, c_border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # SECTION 4: TECH STACK JUSTIFICATION
    story.append(Paragraph("4. Complete Tech Stack & Architectural Justifications", h1_style))
    story.append(Paragraph("Interviewers love to ask: <i>'Why did you choose this technology over alternatives?'</i> Here are your exact justifications:", body_style))

    stack_reasons = [
        ("FastAPI", "High-performance async I/O built on Starlette and Pydantic. Enables native Server-Sent Events (SSE) streaming with low memory overhead, asynchronous worker concurrency, and automatic OpenAPI schema validation."),
        ("PyMuPDF (fitz)", "Over 10x to 25x faster than pure-Python parsers (such as PyPDF2 or PDFMiner). Written in optimized C (MuPDF), it preserves font metadata and layout bounding boxes, enabling accurate multi-column extraction and section header isolation."),
        ("ChromaDB", "Embedded, lightweight vector database providing Hierarchical Navigable Small World (HNSW) indexing. Requires zero external database server setup, persists cleanly to disk via SQLite/hnswlib, and scales effortlessly to 100K+ vectors with sub-15ms queries."),
        ("BM25 (rank-bm25)", "Probabilistic term-weighting algorithm (Okapi BM25). Dense vector embeddings suffer from semantic drift when searching for exact IDs, mathematical terms, or model names. BM25 guarantees precision keyword retrieval."),
        ("LangChain", "Utilized for standardized prompt templating, schema serialization, and modular component interfacing without creating heavy vendor lock-in."),
        ("Pydantic v2", "Strict runtime data validation, automatic serialization, and high-performance Rust-backed JSON parsing for all API schemas, document metadata, and retrieval score objects."),
    ]

    for tool, reason in stack_reasons:
        story.append(Paragraph(f"<b>{tool}:</b> {reason}", bullet_style))
    story.append(Spacer(1, 8))

    # SECTION 5: MATHEMATICAL FORMULATIONS
    story.append(Paragraph("5. Core Mathematical Formulations", h1_style))

    story.append(Paragraph("<b>1. Hybrid Convex Score Combination:</b>", h2_style))
    story.append(Paragraph(
        "To combine dense cosine similarity (S_dense in [0, 1]) and BM25 term scores (normalized to S_bm25 in [0, 1]):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Score_hybrid(d, q) = &alpha; &middot; S_dense(d, q) + (1 &minus; &alpha;) &middot; S_bm25(d, q)</b><br/>"
        "Where &alpha; &isin; [0, 1] is the hybrid weight (default &alpha; = 0.5 balances conceptual semantics with keyword precision).",
        body_style
    ))

    story.append(Paragraph("<b>2. Maximal Marginal Relevance (MMR) Diversity Formulation:</b>", h2_style))
    story.append(Paragraph(
        "To iteratively select a diverse set of passages S from the candidate set R without duplicate context:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>MMR(d) = argmax_{d &isin; R &setminus; S} [ &lambda; &middot; Sim_1(d, Query) &minus; (1 &minus; &lambda;) &middot; max_{s &isin; S} Sim_2(d, s) ]</b><br/>"
        "Where &lambda; &isin; [0, 1] controls diversity tradeoff (default &lambda; = 0.7 gives 70% priority to query relevance and 30% penalty to inter-document redundancy).",
        body_style
    ))
    story.append(Spacer(1, 8))

    # SECTION 6: INTERVIEW QUESTIONS & MODEL ANSWERS
    story.append(Paragraph("6. Top Interview Questions & Comprehensive Answers", h1_style))
    story.append(Paragraph("Study these questions thoroughly &mdash; they reflect senior AI engineering and system design interview rounds.", body_style))

    qa_list = [
        ("Q1: What is Hybrid RAG and why not just use standard Vector Search (Dense Embeddings)?",
         "<b>Answer:</b> Dense vector search projects text into high-dimensional semantic spaces, which is great for finding synonyms or answering paraphrased questions. "
         "However, dense embeddings frequently fail on exact keywords, part numbers, version strings, rare mathematical terms, or author names. "
         "Lexical search (BM25) calculates exact term-frequency and inverse-document-frequency matches. Hybrid RAG combines both worlds: "
         "BM25 captures exact tokens, while ChromaDB captures semantic intent. Blending both ensures near-perfect retrieval recall across all query types."),

        ("Q2: How does Maximal Marginal Relevance (MMR) eliminate duplicate document context?",
         "<b>Answer:</b> In academic papers, the same topic or equation is often repeated in the Abstract, Introduction, and Conclusion. "
         "Standard top-K retrieval picks 3 or 4 copies of that same passage because all of them have high similarity to the query. "
         "MMR prevents this by checking the cosine similarity between candidate passage d and every passage already selected in set S. "
         "If a candidate has high similarity to an already chosen passage, the penalty (1 - lambda) * max(Sim(d, s)) drastically reduces its score, "
         "allowing an informative, non-redundant passage from another section to be selected instead."),

        ("Q3: How does DocuMind track citations and prevent hallucinations?",
         "<b>Answer:</b> We tackle hallucinations at three distinct stages: (1) <i>Ingestion</i>: Every chunk is tagged with immutable metadata (filename, page number, section header). "
         "(2) <i>Prompt Grounding</i>: The system prompt strictly constrains the model to the provided context and mandates the citation anchor format [Doc: filename, Page: p, Section: s]. "
         "(3) <i>Automated Verification</i>: When the answer is generated, the CitationVerifier regex extracts all cited anchors and cross-checks whether the cited page and text actually "
         "exist in the retrieved chunks. If an assertion is uncorroborated, the groundedness score is penalized, providing transparency to the user."),

        ("Q4: Why did you choose PyMuPDF instead of LangChain's default PyPDFLoader?",
         "<b>Answer:</b> PyMuPDF (fitz) is built in C (MuPDF) and is over 10x faster than pure-Python PDF parsers. More importantly, academic research papers typically use multi-column formats. "
         "Standard PyPDFLoader extracts text by raw byte stream, which reads across the page horizontally and scrambles multi-column layouts into incoherent text. "
         "PyMuPDF provides block-level bounding box detection (get_text('blocks')), which correctly sorts vertical text blocks column-by-column, isolating section headers and maintaining reading order."),

        ("Q5: How does Server-Sent Events (SSE) streaming work in your FastAPI backend?",
         "<b>Answer:</b> We implement an asynchronous generator that yields chunked HTTP responses using media_type='text/event-stream'. "
         "First, we emit an 'event: context' payload carrying the retrieved chunks and their score breakdown. "
         "Then, as the LLM yields token deltas asynchronously, we emit 'event: token' carrying each word immediately. "
         "Finally, we emit 'event: citations' with verified anchors and 'event: done' with total latency metrics. "
         "This provides immediate user feedback (Time-To-First-Token < 200ms) while keeping a persistent single HTTP connection open without the overhead of WebSockets."),

        ("Q6: How does DocuMind work without an OpenAI API key?",
         "<b>Answer:</b> In production enterprise environments or restricted air-gapped systems, API keys may not be available immediately. "
         "DocuMind implements a dual-mode provider: (1) When OPENAI_API_KEY is present, it uses OpenAI embeddings and GPT-4o-mini. "
         "(2) When absent, it falls back to a deterministic semantic vectorizer (feature hashing + L2 normalized n-grams) and an offline academic synthesizer "
         "that extracts key declarative propositions from the top MMR chunks. This ensures that CI/CD pipelines, unit tests, and local evaluation run 100% reliably out of the box."),

        ("Q7: What is the Autonomous Research Intelligence mode?",
         "<b>Answer:</b> Standard RAG is single-turn and reactive. The Autonomous Research Agent is proactive: given a broad objective (e.g. 'Compare Transformer vs Mamba across efficiency and memory'), "
         "it uses query decomposition to break the objective into 2 to 4 complementary sub-queries. It executes parallel hybrid retrievals for each sub-query, builds a comparative dimension matrix, "
         "and synthesizes an executive research summary cross-referencing all consulted papers."),

        ("Q8: How would you scale this system to 10 million documents in production?",
         "<b>Answer:</b> To scale to 10M documents: "
         "(1) <i>Vector Layer</i>: Migrate from embedded ChromaDB to a distributed vector engine such as Milvus, Qdrant, or Pinecone with sharded HNSW index partitions. "
         "(2) <i>Lexical Layer</i>: Replace in-memory rank-bm25 with Elasticsearch or OpenSearch cluster using distributed inverted indexes. "
         "(3) <i>Ingestion Pipeline</i>: Decouple PDF extraction and chunking using Celery or Kafka workers with Celery Beat / Ray for parallel document parsing across GPU/CPU nodes. "
         "(4) <i>Caching</i>: Implement a Redis semantic cache on query embeddings to achieve sub-5ms responses on repeated research queries."),
    ]

    for q, a in qa_list:
        story.append(Paragraph(q, qa_q_style))
        story.append(Paragraph(a, qa_a_style))
    story.append(Spacer(1, 8))

    # SECTION 7: ELEVATOR PITCH FOR RECRUITERS & MANAGERS
    story.append(Paragraph("7. 60-Second Interview Elevator Pitch", h1_style))
    pitch_text = (
        "<i>\"In DocuMind, I architected an enterprise Hybrid RAG and autonomous research intelligence system designed specifically for complex scientific PDFs. "
        "Academic papers often break standard RAG systems because multi-column layouts get scrambled, dense embeddings miss exact technical terms, and top-K search gets clogged with duplicate paragraphs. "
        "I solved this by building a layout-aware PyMuPDF ingestion pipeline with page-accurate metadata, combining dense ChromaDB embeddings with lexical BM25 using convex score blending, "
        "and applying Maximal Marginal Relevance (MMR) diversity re-ranking to mathematically eliminate redundant context. "
        "For answer generation, I built a grounded prompt synthesis engine with automated source-page citation verification to stop hallucinations, "
        "and streamed verified tokens via asynchronous FastAPI Server-Sent Events. The entire platform includes a reactive cyber-academic UI with a live Hybrid Search Inspector, "
        "achieves 100% test pass rate across 13 automated tests, and is containerized with Docker and automated GitHub Actions CI.\"</i>"
    )
    story.append(Paragraph(pitch_text, body_style))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {output_path}")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "explanation.pdf"
    build_explanation_pdf(out_file)
