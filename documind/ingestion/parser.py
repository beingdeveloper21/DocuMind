import hashlib
import re
from pathlib import Path
from typing import List, Tuple, Union
import fitz  # PyMuPDF

from documind.core.exceptions import DocumentParsingError
from documind.core.models import DocumentMetadata


class ParsedPage:
    """Represents an extracted page from an academic PDF."""
    def __init__(self, page_number: int, text: str, sections: List[Tuple[str, str]]):
        self.page_number = page_number
        self.text = text
        self.sections = sections  # List of (section_title, section_text)


class PDFDocumentParser:
    """High-performance PDF parser with layout and section awareness using PyMuPDF."""

    HEADER_PATTERN = re.compile(
        r"^(?:(?:\d+\.?)+\s+)?(abstract|introduction|background|related\s+work|methodology|methods|architecture|system\s+design|experiments|results|discussion|evaluation|conclusion|future\s+work|references)\b",
        re.IGNORECASE,
    )

    def parse(self, file_source: Union[str, Path, bytes], filename: str, doc_id: str) -> Tuple[DocumentMetadata, List[ParsedPage]]:
        """
        Parses a PDF from a file path or raw bytes, extracting structured pages and sections.
        """
        try:
            if isinstance(file_source, bytes):
                doc = fitz.open(stream=file_source, filetype="pdf")
                file_size = len(file_source)
                sha256 = hashlib.sha256(file_source).hexdigest()
            else:
                path = Path(file_source)
                if not path.exists():
                    raise DocumentParsingError(f"File not found: {path}")
                content = path.read_bytes()
                file_size = len(content)
                sha256 = hashlib.sha256(content).hexdigest()
                doc = fitz.open(str(path))

            total_pages = doc.page_count
            if total_pages == 0:
                raise DocumentParsingError(f"PDF {filename} contains 0 pages.")

            pages: List[ParsedPage] = []

            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                page_number = page_idx + 1
                blocks = page.get_text("blocks")

                current_section = "General"
                section_blocks: List[Tuple[str, str]] = []
                full_page_text_parts: List[str] = []

                for b in blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(b) >= 5 and b[4]:
                        block_text = b[4].strip()
                        if not block_text:
                            continue

                        # Check if block looks like a section header
                        lines = [line.strip() for line in block_text.split("\n") if line.strip()]
                        if lines:
                            first_line = lines[0]
                            match = self.HEADER_PATTERN.match(first_line)
                            if match or (len(first_line) < 60 and first_line.isupper() and len(lines) == 1):
                                current_section = first_line.strip()

                        full_page_text_parts.append(block_text)
                        section_blocks.append((current_section, block_text))

                page_full_text = "\n\n".join(full_page_text_parts)
                pages.append(ParsedPage(page_number=page_number, text=page_full_text, sections=section_blocks))

            doc.close()

            metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                total_pages=total_pages,
                file_size_bytes=file_size,
                sha256_hash=sha256,
            )

            return metadata, pages

        except Exception as e:
            if isinstance(e, DocumentParsingError):
                raise e
            raise DocumentParsingError(f"Failed to parse PDF document {filename}: {str(e)}") from e
