import uuid
from typing import List
from documind.core.models import DocumentChunk
from documind.ingestion.parser import ParsedPage


class SemanticChunker:
    """
    Recursively chunks extracted PDF pages into granular passages while
    strictly preserving page numbers, section headers, and document provenance.
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, doc_id: str, doc_name: str, pages: List[ParsedPage]) -> List[DocumentChunk]:
        """Splits pages into ordered, metadata-tracked document chunks."""
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages:
            page_number = page.page_number
            # If section blocks are available, chunk by section boundaries to preserve coherence
            if page.sections:
                current_section = ""
                section_accumulator = []

                for sec_name, block_text in page.sections:
                    if sec_name != current_section and section_accumulator:
                        # Flush accumulated section text
                        combined_text = "\n\n".join(section_accumulator)
                        sub_chunks = self._split_text(combined_text)
                        for sc in sub_chunks:
                            chunk_id = f"{doc_id}_p{page_number}_c{global_chunk_idx}_{uuid.uuid4().hex[:6]}"
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=chunk_id,
                                    doc_id=doc_id,
                                    doc_name=doc_name,
                                    page_number=page_number,
                                    section_header=current_section or "General",
                                    content=sc.strip(),
                                    chunk_index=global_chunk_idx,
                                    metadata={
                                        "source": doc_name,
                                        "page": page_number,
                                        "section": current_section or "General",
                                    },
                                )
                            )
                            global_chunk_idx += 1
                        section_accumulator = [block_text]
                        current_section = sec_name
                    else:
                        current_section = sec_name
                        section_accumulator.append(block_text)

                if section_accumulator:
                    combined_text = "\n\n".join(section_accumulator)
                    sub_chunks = self._split_text(combined_text)
                    for sc in sub_chunks:
                        chunk_id = f"{doc_id}_p{page_number}_c{global_chunk_idx}_{uuid.uuid4().hex[:6]}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                doc_name=doc_name,
                                page_number=page_number,
                                section_header=current_section or "General",
                                content=sc.strip(),
                                chunk_index=global_chunk_idx,
                                metadata={
                                    "source": doc_name,
                                    "page": page_number,
                                    "section": current_section or "General",
                                },
                            )
                        )
                        global_chunk_idx += 1
            else:
                # Fallback: chunk whole page text
                sub_chunks = self._split_text(page.text)
                for sc in sub_chunks:
                    chunk_id = f"{doc_id}_p{page_number}_c{global_chunk_idx}_{uuid.uuid4().hex[:6]}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            doc_name=doc_name,
                            page_number=page_number,
                            section_header="General",
                            content=sc.strip(),
                            chunk_index=global_chunk_idx,
                            metadata={
                                "source": doc_name,
                                "page": page_number,
                                "section": "General",
                            },
                        )
                    )
                    global_chunk_idx += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Splits a single body of text respecting paragraph and sentence boundaries."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        results = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                if current_chunk:
                    results.append(current_chunk)
                    # Start next chunk with overlap from the end of current_chunk
                    overlap_seed = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = f"{overlap_seed} {para}".strip()
                else:
                    # Paragraph itself is larger than chunk_size: split on sentence boundaries
                    sentences = para.split(". ")
                    sub_buf = ""
                    for s in sentences:
                        s_clean = s.strip()
                        if not s_clean:
                            continue
                        s_formatted = s_clean if s_clean.endswith(".") else f"{s_clean}."
                        if len(sub_buf) + len(s_formatted) + 1 <= self.chunk_size:
                            sub_buf = f"{sub_buf} {s_formatted}" if sub_buf else s_formatted
                        else:
                            if sub_buf:
                                results.append(sub_buf)
                            sub_buf = s_formatted
                    if sub_buf:
                        current_chunk = sub_buf

        if current_chunk:
            results.append(current_chunk)

        return results
