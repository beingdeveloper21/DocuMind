from pathlib import Path
import pytest
from documind.core.exceptions import DocumentParsingError
from documind.ingestion.chunker import SemanticChunker
from documind.ingestion.parser import PDFDocumentParser


def test_pdf_parsing_sample_file():
    parser = PDFDocumentParser()
    sample_file = Path("sample_data/neural_retrieval_and_adaptive_rag.pdf")
    assert sample_file.exists(), "Sample PDF must exist for ingestion tests"

    metadata, pages = parser.parse(sample_file, filename="neural_retrieval.pdf", doc_id="doc_test_1")

    assert metadata.filename == "neural_retrieval.pdf"
    assert metadata.total_pages == 3
    assert metadata.file_size_bytes > 0
    assert len(metadata.sha256_hash) == 64
    assert len(pages) == 3

    # Check page 1 extraction
    p1 = pages[0]
    assert p1.page_number == 1
    assert "Neural Retrieval" in p1.text or "ABSTRACT" in p1.text

    # Check page 2 extraction
    p2 = pages[1]
    assert p2.page_number == 2
    assert "METHODOLOGY" in p2.text or "Maximal Marginal Relevance" in p2.text


def test_semantic_chunker():
    parser = PDFDocumentParser()
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)

    sample_file = Path("sample_data/neural_retrieval_and_adaptive_rag.pdf")
    metadata, pages = parser.parse(sample_file, filename="neural_retrieval.pdf", doc_id="doc_test_1")

    chunks = chunker.chunk_pages(doc_id=metadata.doc_id, doc_name=metadata.filename, pages=pages)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id == "doc_test_1"
        assert chunk.doc_name == "neural_retrieval.pdf"
        assert chunk.page_number in [1, 2, 3]
        assert len(chunk.content) > 0
        assert chunk.chunk_id.startswith("doc_test_1_")


def test_invalid_pdf_handling():
    parser = PDFDocumentParser()
    with pytest.raises(DocumentParsingError):
        parser.parse(b"not a valid pdf binary stream", filename="corrupt.pdf", doc_id="doc_corrupt")
