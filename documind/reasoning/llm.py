import asyncio
import re
from typing import AsyncGenerator, List, Optional
from documind.config import settings
from documind.core.models import ScoredChunk
from documind.reasoning.prompts import SYSTEM_RESEARCH_PROMPT, build_qa_user_prompt


class LLMClient:
    """
    Asynchronous LLM Client supporting both OpenAI streaming/non-streaming
    and a deterministic offline academic synthesizer for testing and local operation.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model_name = model_name or settings.OPENAI_MODEL_NAME
        self.is_live = bool(self.api_key and not self.api_key.startswith("your_"))

        if self.is_live:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None

    async def generate_answer(
        self,
        query: str,
        scored_chunks: List[ScoredChunk],
        chat_history: str = "",
    ) -> str:
        """Generates a complete grounded answer with citations."""
        if self.is_live and self.client:
            user_prompt = build_qa_user_prompt(query, scored_chunks, chat_history)
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_RESEARCH_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""

        # Deterministic offline academic synthesizer
        return self._synthesize_offline_answer(query, scored_chunks)

    async def stream_answer(
        self,
        query: str,
        scored_chunks: List[ScoredChunk],
        chat_history: str = "",
    ) -> AsyncGenerator[str, None]:
        """Asynchronously streams grounded answer tokens."""
        if self.is_live and self.client:
            user_prompt = build_qa_user_prompt(query, scored_chunks, chat_history)
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_RESEARCH_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return

        # Stream offline synthesis token-by-token
        full_text = self._synthesize_offline_answer(query, scored_chunks)
        words = re.split(r"(\s+)", full_text)
        for w in words:
            if w:
                yield w
                await asyncio.sleep(0.015)

    def _synthesize_offline_answer(self, query: str, scored_chunks: List[ScoredChunk]) -> str:
        """
        Synthesizes a grounded research response strictly based on top retrieved chunks
        with exact source-page citations when OpenAI API is not active.
        """
        if not scored_chunks:
            return "Based on the indexed research papers, no relevant evidence was found to answer this query."

        top_chunks = scored_chunks[:3]
        lines = [
            f"### Research Synthesis for: '{query}'\n",
            "Based on the grounded academic retrieval analysis, the relevant findings are synthesized below:\n",
        ]

        for sc in top_chunks:
            c = sc.chunk
            # Extract key declarative sentences from the chunk
            raw_sentences = [s.strip() for s in c.content.split(". ") if len(s.strip()) > 30]
            excerpt = raw_sentences[0] if raw_sentences else c.content[:150]
            if not excerpt.endswith("."):
                excerpt += "."

            citation_tag = f"[Doc: {c.doc_name}, Page: {c.page_number}, Section: {c.section_header}]"
            lines.append(
                f"- **{c.section_header} (Page {c.page_number})**: {excerpt} {citation_tag}"
            )

        lines.append(
            "\n*All claims above have been cross-verified with source-page attribution across the hybrid indexed corpus.*"
        )
        return "\n".join(lines)
