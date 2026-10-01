"""Base class for all RAG pipelines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import AgentResponse, RAGResponse


class BaseRAG(ABC):
    """
    Abstract Base Class for all RAG architectures.

    Encapsulates core document ingestion, chunking, embedding generation,
    vector storage, and similarity retrieval primitives.
    """

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
        self.chunker = chunker or RecursiveCharacterChunker()

    def ingest_text(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk, embed, and store document text into the vector database."""
        if not text.strip():
            return []

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
        """Read and ingest a text or markdown file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(text=text, source=path.name, metadata=metadata)

    def ingest_directory(
        self,
        dir_path: Path | str,
        glob_pattern: str = "*.txt",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recursively scan and ingest all matching files in a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        added: list[dict[str, Any]] = []
        for file in sorted(path.rglob(glob_pattern)):
            if file.is_file():
                docs = self.ingest_file(file, metadata=metadata)
                added.extend(docs)
        return added

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Find the top-k most relevant chunks for a query vector."""
        query_vector = self.embedding_model.embed_text(query)
        return self.vector_store.search(query_vector=query_vector, top_k=top_k, **kwargs)

    def format_context(
        self,
        search_results: list[dict[str, Any]],
    ) -> str:
        """Format search results into a clean context string for LLM prompting."""
        blocks: list[str] = []
        for result in search_results:
            document = result.get("document", {})
            source = document.get("source", "unknown")
            chunk_id = document.get("chunk_id", "")
            text = document.get("text", "")
            blocks.append(f"Source: {source}#{chunk_id}\n{text}")
        return "\n\n---\n\n".join(blocks)

    @abstractmethod
    def execute(self, question: str, **kwargs: Any) -> RAGResponse | AgentResponse:
        """Execute end-to-end question answering pipeline."""
        raise NotImplementedError

    def query(self, question: str, **kwargs: Any) -> RAGResponse | AgentResponse:
        """Convenience alias for execute."""
        return self.execute(question, **kwargs)


__all__ = ["BaseRAG"]
