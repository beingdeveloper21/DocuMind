import json
import time
from typing import List, Optional
from documind.core.models import (
    AutonomousResearchReport,
    Citation,
    ResearchSubQuery,
    ScoredChunk,
)
from documind.reasoning.citation_verifier import CitationVerifier
from documind.reasoning.llm import LLMClient
from documind.reasoning.prompts import SYSTEM_DECOMPOSITION_PROMPT
from documind.retrieval.hybrid_retriever import HybridRetriever


class AutonomousResearchAgent:
    """
    Autonomous Research Intelligence Engine.
    Decomposes multi-faceted research hypotheses into focused sub-queries,
    orchestrates hybrid retrieval across papers, synthesizes comparative insights,
    and validates source-page attribution.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: Optional[LLMClient] = None,
        citation_verifier: Optional[CitationVerifier] = None,
    ):
        self.retriever = retriever
        self.llm = llm_client or LLMClient()
        self.verifier = citation_verifier or CitationVerifier()

    async def execute_research(
        self,
        objective: str,
        doc_ids: Optional[List[str]] = None,
        sub_query_limit: int = 3,
    ) -> AutonomousResearchReport:
        """Executes the full autonomous research intelligence lifecycle."""
        start_time = time.perf_counter()

        # Phase 1: Query Decomposition
        sub_queries = await self._decompose_objective(objective, limit=sub_query_limit)

        # Phase 2: Parallel Hybrid Retrieval & Sub-Analysis
        papers_consulted_set = set()
        all_retrieved_chunks: List[ScoredChunk] = []

        for sq in sub_queries:
            chunks = self.retriever.retrieve(
                query=sq.sub_query,
                top_k=8,
                final_k=3,
                doc_ids=doc_ids,
                apply_mmr=True,
            )
            sq.retrieved_evidence = chunks
            for sc in chunks:
                papers_consulted_set.add(sc.chunk.doc_name)
                all_retrieved_chunks.append(sc)

            # Generate focused sub-answer
            sub_ans = await self.llm.generate_answer(
                query=sq.sub_query,
                scored_chunks=chunks,
            )
            sq.sub_answer = sub_ans

        # Phase 3: Executive Synthesis & Comparative Matrix
        comparative_matrix = self._build_comparative_matrix(sub_queries)

        synthesis_prompt = (
            f"Overarching Research Objective: {objective}\n\n"
            f"Findings from Decomposed Sub-Investigations:\n"
            + "\n".join([f"Sub-Query: {sq.sub_query}\nAnswer: {sq.sub_answer}" for sq in sub_queries])
            + "\n\nSynthesize an authoritative, coherent Executive Research Summary linking these findings with citations."
        )

        executive_summary = await self.llm.generate_answer(
            query=synthesis_prompt,
            scored_chunks=all_retrieved_chunks,
        )

        # Phase 4: Citation Extraction & Validation
        citations, _ = self.verifier.verify_and_extract(executive_summary, all_retrieved_chunks)

        latency = round((time.perf_counter() - start_time) * 1000.0, 2)

        return AutonomousResearchReport(
            objective=objective,
            decomposed_queries=sub_queries,
            comparative_matrix=comparative_matrix,
            executive_synthesis=executive_summary,
            citations=citations,
            papers_consulted=sorted(list(papers_consulted_set)),
            latency_ms=latency,
        )

    async def _decompose_objective(self, objective: str, limit: int = 3) -> List[ResearchSubQuery]:
        """Decomposes a broad research goal into complementary sub-queries."""
        if self.llm.is_live and self.llm.client:
            try:
                response = await self.llm.client.chat.completions.create(
                    model=self.llm.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_DECOMPOSITION_PROMPT},
                        {"role": "user", "content": f"Decompose this research objective: {objective}"},
                    ],
                    temperature=0.3,
                )
                raw_json = response.choices[0].message.content or "[]"
                # Parse JSON array
                data = json.loads(raw_json.strip("` \n").replace("json\n", ""))
                sub_queries = [
                    ResearchSubQuery(
                        sub_query=item.get("sub_query", objective),
                        rationale=item.get("rationale", "Domain decomposition"),
                    )
                    for item in data[:limit]
                ]
                if sub_queries:
                    return sub_queries
            except Exception:
                pass

        # Deterministic academic decomposition heuristics
        return [
            ResearchSubQuery(
                sub_query=f"Core methodology and architectural design regarding {objective}",
                rationale="Analyze foundational system paradigms and structural mechanics.",
            ),
            ResearchSubQuery(
                sub_query=f"Empirical benchmarks, ablation studies, and performance results in {objective}",
                rationale="Evaluate quantitative metrics, baseline comparisons, and trade-offs.",
            ),
            ResearchSubQuery(
                sub_query=f"Limitations, failure modes, and theoretical constraints of {objective}",
                rationale="Identify critical edge cases and practical bottlenecks.",
            ),
        ][:limit]

    def _build_comparative_matrix(self, sub_queries: List[ResearchSubQuery]) -> dict:
        """Constructs a comparative dimension map across sub-query findings."""
        matrix = {}
        for idx, sq in enumerate(sub_queries, start=1):
            key = f"Dimension {idx}: {sq.sub_query[:40]}..."
            matrix[key] = {
                "rationale": sq.rationale,
                "evidence_chunks_count": len(sq.retrieved_evidence),
                "summary": sq.sub_answer[:200] + "..." if len(sq.sub_answer) > 200 else sq.sub_answer,
            }
        return matrix
