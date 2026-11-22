from typing import List
from documind.core.models import ScoredChunk


SYSTEM_RESEARCH_PROMPT = """You are DocuMind, an enterprise autonomous research intelligence and hybrid RAG system specialized in academic scientific literature.

Core Grounding Directives:
1. Grounding Integrity: Answer the user's research question strictly using the provided context passages. Do not introduce outside speculative claims.
2. Citation Tracking: For EVERY factual statement, data point, or architectural detail, you MUST cite the exact source document, page number, and section using the explicit format:
   [Doc: <filename>, Page: <page_number>, Section: <section_name>]
3. Rigor & Precision: Maintain high academic clarity. Differentiate between empirical findings, theoretical bounds, and methodology choices.
4. Transparency: If the retrieved documents do not contain sufficient evidence to answer the question, explicitly state the limitation rather than guessing.
"""

SYSTEM_DECOMPOSITION_PROMPT = """You are an autonomous research query planner for academic literature.
Given a complex research objective, decompose it into 2 to 4 precise, complementary sub-queries suitable for hybrid lexical and vector retrieval across academic papers.

Format your response as a valid JSON list of objects:
[
  {
    "sub_query": "specific targeted research question",
    "rationale": "why this sub-query is essential to solving the overarching objective"
  }
]
Only return the raw JSON list.
"""


def format_context_with_citations(scored_chunks: List[ScoredChunk]) -> str:
    """Formats retrieved chunks with clear provenance headers for LLM prompt injection."""
    if not scored_chunks:
        return "No relevant academic passages retrieved."

    formatted_passages = []
    for idx, sc in enumerate(scored_chunks, start=1):
        c = sc.chunk
        header = (
            f"--- Context Passage {idx} ---\n"
            f"Document: {c.doc_name}\n"
            f"Page: {c.page_number}\n"
            f"Section: {c.section_header}\n"
            f"Retrieval Score: Hybrid={sc.hybrid_score:.3f} (Dense={sc.dense_score:.3f}, BM25={sc.bm25_score:.3f}, MMR Rank={sc.mmr_rank})\n"
            f"Content:\n{c.content}\n"
        )
        formatted_passages.append(header)

    return "\n".join(formatted_passages)


def build_qa_user_prompt(query: str, scored_chunks: List[ScoredChunk], chat_history: str = "") -> str:
    """Builds the final user prompt containing context, conversational memory, and query."""
    context_str = format_context_with_citations(scored_chunks)

    history_block = f"Prior Conversation History:\n{chat_history}\n\n" if chat_history else ""

    prompt = (
        f"{history_block}"
        f"Retrieved Academic Context:\n{context_str}\n\n"
        f"Research Question: {query}\n\n"
        f"Provide a verified, comprehensive answer with source-page citations [Doc: <filename>, Page: <p>, Section: <s>]:"
    )
    return prompt
