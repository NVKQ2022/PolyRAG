"""Standard Retrieve-then-Read Naive RAG pipeline."""

import time
from typing import Any

from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import RAGResponse
from polyrag.pipelines.base import BaseRAG


class NaiveRAG(BaseRAG):
    """
    Standard retrieve-then-read RAG pipeline.

    Inherits document ingestion, chunking, and similarity search from BaseRAG
    and implements a 1-shot retrieval and prompt generation workflow.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        llm_client: BaseLLMClient | None = None,
        chunker: BaseChunker | None = None,
    ) -> None:
        super().__init__(
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm_client=llm_client,
            chunker=chunker,
        )

    def execute(
        self,
        question: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RAGResponse:
        """Perform end-to-end RAG retrieval and answer generation."""
        start_time = time.time()
        results = self.retrieve(question, top_k=top_k)
        context = self.format_context(results)

        if self.llm_client is None:
            # Return context retrieval without generation if no LLM configured
            took_ms = int((time.time() - start_time) * 1000)
            return RAGResponse(
                question=question,
                answer="",
                context=context,
                sources=[r.get("document", {}) for r in results],
                took_ms=took_ms,
                confidence=1.0 if results else 0.0,
                reasoning_summary="Retrieval completed without LLM generation.",
                llm_calls=0,
            )

        prompt = (
            "Use the following context to answer the question. "
            "If the answer is not in the context, state that the context lacks sufficient information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer with citations where possible:"
        )

        answer = self.llm_client.complete(prompt)
        took_ms = int((time.time() - start_time) * 1000)

        sources = [result.get("document", {}) for result in results]

        return RAGResponse(
            question=question,
            answer=answer,
            context=context,
            sources=sources,
            took_ms=took_ms,
            confidence=0.85 if results else 0.2,
            reasoning_summary=f"Retrieved {len(results)} chunks and generated answer.",
            llm_calls=1,
        )


__all__ = ["NaiveRAG"]
