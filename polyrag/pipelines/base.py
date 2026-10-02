"""Base class for all RAG pipelines built on LangChain primitives."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter

from polyrag.chunkers import resolve_chunker
from polyrag.core.models import AgentResponse, Document, RAGResponse
from polyrag.embeddings import resolve_embedding_model
from polyrag.llms import resolve_llm_client
from polyrag.vector_stores import resolve_vector_store


class BaseRAG(ABC):
    """
    Abstract Base Class for all RAG architectures built on LangChain.

    Encapsulates document ingestion, chunking via TextSplitter, vector storage
    via VectorStore, and similarity retrieval.
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
        target_emb = embedding if embedding is not None else embedding_model
        self.embedding_model = resolve_embedding_model(target_emb)
        self.vector_store = resolve_vector_store(vector_store, embedding=self.embedding_model)

        target_llm = chat_model if chat_model is not None else (llm if llm is not None else llm_client)
        self.llm_client = resolve_llm_client(target_llm) if target_llm is not None else None
        self.chat_model = self.llm_client
        self.chunker = resolve_chunker(chunker)

    def ingest_text(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
        chunker: TextSplitter | str | None = None,
        embedding_model: Embeddings | str | Any | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk, embed, and store document text into the vector database."""
        if not text or not text.strip():
            return []

        active_chunker = resolve_chunker(chunker) if chunker is not None else self.chunker
        if hasattr(active_chunker, "split_text"):
            chunks = active_chunker.split_text(text)
        elif hasattr(active_chunker, "chunk"):
            chunks = active_chunker.chunk(text)
        else:
            chunks = [text]

        if not chunks:
            return []

        active_embedding = (
            resolve_embedding_model(embedding_model)
            if embedding_model is not None
            else self.embedding_model
        )

        if hasattr(active_embedding, "embed_documents"):
            vectors = active_embedding.embed_documents(chunks)
        elif hasattr(active_embedding, "embed_batch"):
            vectors = active_embedding.embed_batch(chunks)
        elif hasattr(active_embedding, "embed_text"):
            vectors = [active_embedding.embed_text(c) for c in chunks]
        else:
            vectors = [[0.0] * 384 for _ in chunks]

        lc_docs: list[Document] = []
        doc_dicts: list[dict[str, Any]] = []
        for cid, chunk_text in enumerate(chunks):
            doc_meta = {
                "source": source,
                "chunk_id": cid,
                **(metadata or {}),
            }
            lc_docs.append(Document(page_content=chunk_text, metadata=doc_meta))
            doc_dicts.append({
                "text": chunk_text,
                "page_content": chunk_text,
                **doc_meta,
            })

        # Add to vector store
        try:
            self.vector_store.add_documents(vectors=vectors, documents=doc_dicts)
        except (TypeError, AttributeError):
            try:
                self.vector_store.add_documents(lc_docs)
            except Exception:
                self.vector_store.add_texts(
                    texts=[d["text"] for d in doc_dicts],
                    metadatas=doc_dicts,
                )

        return doc_dicts

    def ingest_file(
        self,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None,
        chunker: TextSplitter | str | None = None,
        embedding_model: Embeddings | str | Any | None = None,
    ) -> list[dict[str, Any]]:
        """Read and ingest a text or markdown file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(
            text=text,
            source=path.name,
            metadata=metadata,
            chunker=chunker,
            embedding_model=embedding_model,
        )

    def ingest_directory(
        self,
        dir_path: Path | str,
        glob_pattern: str = "*.txt",
        metadata: dict[str, Any] | None = None,
        chunker: TextSplitter | str | None = None,
        embedding_model: Embeddings | str | Any | None = None,
    ) -> list[dict[str, Any]]:
        """Recursively scan and ingest all matching files in a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        added: list[dict[str, Any]] = []
        for file in sorted(path.rglob(glob_pattern)):
            if file.is_file():
                docs = self.ingest_file(
                    file,
                    metadata=metadata,
                    chunker=chunker,
                    embedding_model=embedding_model,
                )
                added.extend(docs)
        return added

    def ingest_documents(
        self,
        documents: Iterable[Any],
        metadata: dict[str, Any] | None = None,
        chunker: TextSplitter | str | None = None,
        embedding_model: Embeddings | str | Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ingest an iterable, generator, or list of documents.

        Supports:
        - LangChain Document objects (has 'page_content' and 'metadata')
        - PolyRAG Document models (has 'text' and 'metadata')
        - Dictionaries ({"text": ..., "source": ...})
        - Raw strings
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

            chunks = self.ingest_text(
                text=text,
                source=source,
                metadata=doc_meta,
                chunker=chunker,
                embedding_model=embedding_model,
            )
            added.extend(chunks)

        return added

    def ingest_langchain_loader(
        self,
        loader: Any,
        metadata: dict[str, Any] | None = None,
        chunker: TextSplitter | str | None = None,
        embedding_model: Embeddings | str | Any | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest documents from any LangChain DocumentLoader."""
        if hasattr(loader, "lazy_load"):
            docs = loader.lazy_load()
        elif hasattr(loader, "load"):
            docs = loader.load()
        else:
            raise TypeError("Provided loader object must implement 'lazy_load()' or 'load()'.")

        return self.ingest_documents(
            docs,
            metadata=metadata,
            chunker=chunker,
            embedding_model=embedding_model,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        embedding_model: Embeddings | str | Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Find the top-k most relevant chunks for a query string."""
        active_embedding = (
            resolve_embedding_model(embedding_model)
            if embedding_model is not None
            else self.embedding_model
        )

        if hasattr(active_embedding, "embed_query"):
            query_vector = active_embedding.embed_query(query)
        elif hasattr(active_embedding, "embed_text"):
            query_vector = active_embedding.embed_text(query)
        else:
            query_vector = [0.0] * 384

        # 1. Custom or PolyRAG vector store with search()
        if hasattr(self.vector_store, "search"):
            try:
                return self.vector_store.search(query_vector=query_vector, top_k=top_k, **kwargs)
            except Exception:
                pass

        # 2. Native LangChain VectorStore similarity_search_with_score_by_vector
        if hasattr(self.vector_store, "similarity_search_with_score_by_vector"):
            try:
                hits = self.vector_store.similarity_search_with_score_by_vector(query_vector, k=top_k, **kwargs)
                results: list[dict[str, Any]] = []
                for hit in hits:
                    if isinstance(hit, tuple):
                        doc, score = hit
                    else:
                        doc, score = hit, 1.0

                    s_float = float(score)
                    results.append({
                        "score": s_float,
                        "distance": 1.0 - s_float if s_float <= 1.0 else s_float,
                        "document": {
                            "text": doc.page_content,
                            "page_content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            **doc.metadata,
                        },
                    })
                return results
            except Exception:
                pass

        # 3. Native LangChain VectorStore similarity_search_with_score
        if hasattr(self.vector_store, "similarity_search_with_score"):
            try:
                hits = self.vector_store.similarity_search_with_score(query, k=top_k, **kwargs)
                results = []
                for hit in hits:
                    if isinstance(hit, tuple):
                        doc, score = hit
                    else:
                        doc, score = hit, 1.0

                    s_float = float(score)
                    results.append({
                        "score": s_float,
                        "distance": 1.0 - s_float if s_float <= 1.0 else s_float,
                        "document": {
                            "text": doc.page_content,
                            "page_content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            **doc.metadata,
                        },
                    })
                return results
            except Exception:
                pass

        # 4. Native LangChain VectorStore similarity_search
        if hasattr(self.vector_store, "similarity_search"):
            try:
                hits = self.vector_store.similarity_search(query, k=top_k, **kwargs)
                results = []
                for rank, doc in enumerate(hits):
                    results.append({
                        "score": 1.0 / (1.0 + rank),
                        "distance": float(rank),
                        "document": {
                            "text": doc.page_content,
                            "page_content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            **doc.metadata,
                        },
                    })
                return results
            except Exception:
                pass

        return []

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
            text = document.get("text", document.get("page_content", ""))
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
