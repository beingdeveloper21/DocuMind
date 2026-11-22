"""
Reasoning and grounded prompt synthesis module with citation tracking.
"""

from documind.reasoning.prompts import (
    SYSTEM_RESEARCH_PROMPT,
    SYSTEM_DECOMPOSITION_PROMPT,
    format_context_with_citations,
    build_qa_user_prompt,
)
from documind.reasoning.citation_verifier import CitationVerifier
from documind.reasoning.llm import LLMClient
from documind.reasoning.research_agent import AutonomousResearchAgent

__all__ = [
    "SYSTEM_RESEARCH_PROMPT",
    "SYSTEM_DECOMPOSITION_PROMPT",
    "format_context_with_citations",
    "build_qa_user_prompt",
    "CitationVerifier",
    "LLMClient",
    "AutonomousResearchAgent",
]
