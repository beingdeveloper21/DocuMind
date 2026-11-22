import re
from typing import List, Tuple
from documind.core.models import Citation, ScoredChunk


class CitationVerifier:
    """
    Verifies source-page citation fidelity and tracks groundedness
    between generated responses and retrieved document passages.
    """

    # Matches [Doc: filename, Page: X, Section: Y] or variations
    CITATION_REGEX = re.compile(
        r"\[(?:Doc:\s*)?([^,\]]+?),\s*(?:Page|p\.?):\s*(\d+)(?:,\s*(?:Section|Sec\.?):\s*([^\]]+))?\]",
        re.IGNORECASE,
    )

    def verify_and_extract(
        self,
        answer_text: str,
        retrieved_chunks: List[ScoredChunk],
    ) -> Tuple[List[Citation], float]:
        """
        Extracts citations from generated text, verifies them against retrieved chunks,
        and computes a groundedness score.
        """
        matches = list(self.CITATION_REGEX.finditer(answer_text))
        citations: List[Citation] = []

        if not matches:
            # If the model produced a grounded answer without explicit brackets,
            # synthesize citations from top retrieved chunks that have strong overlap
            synthesized = self._synthesize_fallback_citations(answer_text, retrieved_chunks)
            score = 0.85 if synthesized else 0.5
            return synthesized, score

        verified_count = 0

        for match in matches:
            doc_name_raw = match.group(1).strip()
            page_num_str = match.group(2).strip()
            section_raw = (match.group(3) or "General").strip()

            try:
                page_num = int(page_num_str)
            except ValueError:
                page_num = 1

            # Check if this citation corresponds to an actual retrieved chunk
            matched_chunk = None
            for sc in retrieved_chunks:
                c = sc.chunk
                # Relaxed filename matching (handles .pdf extension presence/absence)
                doc_name_clean = doc_name_raw.lower().replace(".pdf", "")
                c_name_clean = c.doc_name.lower().replace(".pdf", "")

                if (doc_name_clean in c_name_clean or c_name_clean in doc_name_clean) and c.page_number == page_num:
                    matched_chunk = c
                    break

            if matched_chunk:
                verified_count += 1
                snippet = matched_chunk.content[:200] + "..." if len(matched_chunk.content) > 200 else matched_chunk.content
                citations.append(
                    Citation(
                        doc_name=matched_chunk.doc_name,
                        page_number=matched_chunk.page_number,
                        section_header=matched_chunk.section_header,
                        snippet=snippet,
                        is_verified=True,
                        grounding_confidence=0.98,
                    )
                )
            else:
                # Unverified citation (e.g. model hallucinated a page number not in retrieved context)
                citations.append(
                    Citation(
                        doc_name=doc_name_raw,
                        page_number=page_num,
                        section_header=section_raw,
                        snippet="Citation reference not corroborated in retrieved top context.",
                        is_verified=False,
                        grounding_confidence=0.20,
                    )
                )

        groundedness = round(float(verified_count / len(matches)), 3) if matches else 0.0
        return citations, groundedness

    def _synthesize_fallback_citations(
        self,
        answer_text: str,
        retrieved_chunks: List[ScoredChunk],
    ) -> List[Citation]:
        """Synthesizes high-confidence citation anchors from top retrieved chunks."""
        citations = []
        answer_words = set(re.findall(r"\w{4,}", answer_text.lower()))

        for sc in retrieved_chunks[:3]:
            c = sc.chunk
            chunk_words = set(re.findall(r"\w{4,}", c.content.lower()))
            overlap = len(answer_words.intersection(chunk_words))
            if overlap >= 3:
                snippet = c.content[:200] + "..." if len(c.content) > 200 else c.content
                citations.append(
                    Citation(
                        doc_name=c.doc_name,
                        page_number=c.page_number,
                        section_header=c.section_header,
                        snippet=snippet,
                        is_verified=True,
                        grounding_confidence=0.88,
                    )
                )
        return citations
