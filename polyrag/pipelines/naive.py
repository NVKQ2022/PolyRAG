"""Standard Retrieve-then-Read Naive RAG pipeline built on LangChain primitives."""

import time
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter

from polyrag.core.models import RAGResponse
from polyrag.pipelines.base import BaseRAG


def _generate_text(llm: Any, prompt: str, **kwargs: Any) -> str:
    """Generate completion text from any LangChain ChatModel, LLM, or callable."""
    if hasattr(llm, "invoke"):
        res = llm.invoke(prompt, **kwargs)
        if hasattr(res, "content"):
            return str(res.content).strip()
        return str(res).strip()
    if hasattr(llm, "complete"):
        return str(llm.complete(prompt, **kwargs)).strip()
    if callable(llm):
        return str(llm(prompt)).strip()
    return str(llm).strip()


class NaiveRAG(BaseRAG):
    """
    Standard retrieve-then-read RAG pipeline.

    Inherits document ingestion, chunking, and similarity search from BaseRAG
    and implements a 1-shot retrieval and prompt generation workflow.
    """

    def __init__(
        self,
        embedding_model: Embeddings | str | Any | None = None,
        vector_store: VectorStore | None = None,
        llm_client: BaseChatModel | str | Any | None = None,
        chunker: TextSplitter | str | None = None,
        embedding: Embeddings | str | Any | None = None,
        chat_model: BaseChatModel | str | Any | None = None,
        llm: BaseChatModel | str | Any | None = None,
    ) -> None:
        super().__init__(
            embedding_model=embedding if embedding is not None else embedding_model,
            vector_store=vector_store,
            llm_client=llm_client,
            chunker=chunker,
            chat_model=chat_model,
            llm=llm,
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

        answer = _generate_text(self.llm_client, prompt, **kwargs)
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
