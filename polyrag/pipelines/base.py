"""Base class for all RAG pipelines."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
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

    def ingest_documents(
        self,
        documents: Iterable[Any],
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ingest an iterable, generator, or list of documents.

        Supports:
        - LangChain Document objects (has 'page_content' and 'metadata')
        - PolyRAG Document models (has 'text' and 'metadata')
        - Dictionaries ({"text": ..., "source": ...})
        - Raw strings

        Args:
            documents: An iterable of documents (e.g. from loader.lazy_load()).
            metadata: Optional global metadata to attach to all ingested documents.

        Returns:
            List of indexed chunk dictionaries.
        """
        added: list[dict[str, Any]] = []
        for item in documents:
            if hasattr(item, "page_content"):
                text = str(item.page_content)
                doc_meta = dict(getattr(item, "metadata", {}))
                doc_id = getattr(item, "id", None)
                if doc_id:
                    doc_meta["_id"] = doc_id
                source = doc_meta.get("source", "external_doc")
            elif hasattr(item, "text"):
                text = str(item.text)
                doc_meta = dict(getattr(item, "metadata", {}))
                if getattr(item, "doc_id", None):
                    doc_meta["_id"] = item.doc_id
                source = getattr(item, "source", "document")
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("page_content") or "")
                doc_meta = {k: v for k, v in item.items() if k not in ("text", "page_content")}
                source = str(item.get("source", "document"))
            else:
                text = str(item)
                doc_meta = {}
                source = "document"

            if metadata:
                doc_meta.update(metadata)

            chunks = self.ingest_text(text=text, source=source, metadata=doc_meta)
            added.extend(chunks)

        return added

    def ingest_langchain_loader(
        self,
        loader: Any,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ingest documents from any LangChain DocumentLoader (e.g. PyPDFLoader, CSVLoader, WebBaseLoader).

        Streams memory-efficiently using loader.lazy_load() when available, falling back to loader.load().

        Args:
            loader: A LangChain DocumentLoader instance.
            metadata: Optional metadata to merge into all loaded documents.

        Returns:
            List of indexed chunk dictionaries.
        """
        if hasattr(loader, "lazy_load"):
            docs = loader.lazy_load()
        elif hasattr(loader, "load"):
            docs = loader.load()
        else:
            raise TypeError("Provided loader object must implement 'lazy_load()' or 'load()'.")

        return self.ingest_documents(docs, metadata=metadata)

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
