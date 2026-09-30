"""Standard Retrieve-then-Read Naive RAG pipeline."""

from pathlib import Path
import time
from typing import Any

from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import RAGResponse


class NaiveRAG:
    """Standard retrieve-then-read RAG pipeline."""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        llm_client: BaseLLMClient | None = None,
        chunker: BaseChunker | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.chunker = chunker

    def ingest_text(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk, embed, and store document text into the vector database."""
        if not text.strip():
            return []
        if self.chunker is None:
            from polyrag.chunkers.fixed_size import FixedSizeChunker
            self.chunker = FixedSizeChunker()

        chunks = self.chunker.chunk(text)
        if not chunks:
            return []

        vectors = self.embedding_model.embed_batch(chunks)
        documents = []
        for cid, chunk_text in enumerate(chunks):
            doc = {
                "text": chunk_text,
                "source": source,
                "chunk_id": cid,
            }
            if metadata:
                doc.update(metadata)
            documents.append(doc)

        self.vector_store.add_documents(vectors=vectors, documents=documents)
        return documents

    def ingest_file(
        self,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Read and ingest a text file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(text=text, source=path.name, metadata=metadata)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Find the top-k most relevant chunks for a query."""
        query_vector = self.embedding_model.embed_text(query)
        return self.vector_store.search(query_vector=query_vector, top_k=top_k)

    def format_context(
        self,
        search_results: list[dict[str, Any]],
    ) -> str:
        """Format search results into a clean context string."""
        blocks: list[str] = []
        for result in search_results:
            document = result.get("document", {})
            source = document.get("source", "unknown")
            chunk_id = document.get("chunk_id", "")
            text = document.get("text", "")
            blocks.append(f"Source: {source}#{chunk_id}\n{text}")
        return "\n\n---\n\n".join(blocks)

    def execute(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """Execute the retrieve-then-read pipeline."""
        start_time = time.time()
        results = self.retrieve(question, top_k=top_k)
        context = self.format_context(results)

        if not self.llm_client:
            took_ms = int((time.time() - start_time) * 1000)
            return RAGResponse(
                question=question,
                answer="",
                context=context,
                sources=[r.get("document", {}) for r in results],
                took_ms=took_ms,
                confidence=0.5,
                reasoning_summary="Retrieved evidence without LLM generation.",
                llm_calls=0,
            )

        prompt = (
            "Use the following context to answer the question. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        answer = self.llm_client.complete(prompt)
        took_ms = int((time.time() - start_time) * 1000)

        return RAGResponse(
            question=question,
            answer=answer,
            context=context,
            sources=[r.get("document", {}) for r in results],
            took_ms=took_ms,
            confidence=0.85,
            reasoning_summary=f"Retrieved {len(results)} chunks and generated answer using {self.llm_client.model_name}.",
            llm_calls=1,
        )

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Convenience method returning a dictionary."""
        return self.execute(question, top_k=top_k).to_dict()
