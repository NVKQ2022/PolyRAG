"""
Application context and pipeline factory for PolyRAG.

PolyRAG serves as the central setup orchestrator and pipeline factory:
1. Configures environment defaults (LLM, Embeddings, Vector Store, Chunker).
2. Manages shared document ingestion across text, files, and directories.
3. Manufactures specialized RAG pipelines (NaiveRAG, AdvancedRAG, AgenticRAG, ReActAgent)
   that operate on the configured models and populated vector store.
"""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
from typing import Any

from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.container import Container
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import RAGResponse
from polyrag.embeddings.sentence_transformers import SentenceTransformerEmbedding
from polyrag.llms.openai import OpenAILLM
from polyrag.pipelines.advanced import AdvancedRAG
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent
from polyrag.vector_stores.chroma import ChromaVectorStore
from polyrag.vector_stores.memory import InMemoryVectorStore


class PolyRAG:
    """
    Central application context and pipeline factory for PolyRAG.

    Coordinates component configuration, shared document ingestion, and
    pipeline instantiation across Naive, Advanced, Agentic, and ReAct strategies.
    """

    def __init__(
        self,
        chunker: BaseChunker | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        vector_store: BaseVectorStore | None = None,
        llm_client: BaseLLMClient | None = None,
        container: Container | None = None,
    ) -> None:
        if container is not None:
            self.container = container
            self.chunker = container.resolve(BaseChunker) if container.is_registered(BaseChunker) else (chunker or RecursiveCharacterChunker())
            self.embedding_model = container.resolve(BaseEmbeddingModel) if container.is_registered(BaseEmbeddingModel) else (embedding_model or SentenceTransformerEmbedding())
            self.vector_store = container.resolve(BaseVectorStore) if container.is_registered(BaseVectorStore) else (vector_store or InMemoryVectorStore())
            self.llm_client = container.resolve(BaseLLMClient) if container.is_registered(BaseLLMClient) else (llm_client or OpenAILLM())
        else:
            self.chunker = chunker or RecursiveCharacterChunker()
            self.embedding_model = embedding_model or SentenceTransformerEmbedding()
            self.vector_store = vector_store or InMemoryVectorStore()
            self.llm_client = llm_client or OpenAILLM()
            self.container = Container.create(
                chunker=self.chunker,
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
                llm_client=self.llm_client,
            )

        # Internal default naive pipeline for shared ingestion and convenience queries
        self._default_pipeline = NaiveRAG(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            chunker=self.chunker,
        )

        # Backward compatibility attribute aliases
        self.chunking_service = self.chunker
        self.embedding_service = self.embedding_model
        self.vector_db = self.vector_store
        self.client = getattr(self.llm_client, "client", self.llm_client)

    # -------------------------------------------------------------------------
    # Factory Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        persist_dir: str | Path | None = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> PolyRAG:
        """Initialize PolyRAG application context from environment variables."""
        container = Container.from_env(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        return cls(container=container)

    @classmethod
    def create(
        cls,
        model_name: str = "gpt-4o-mini",
        embedding_model: BaseEmbeddingModel | str = "all-MiniLM-L6-v2",
        persist_dir: str | Path | None = None,
        collection_name: str = "documents",
        vector_store: BaseVectorStore | None = None,
        llm_client: BaseLLMClient | None = None,
        chunker: BaseChunker | None = None,
    ) -> PolyRAG:
        """Construct a PolyRAG application context with sensible defaults."""
        llm = llm_client or OpenAILLM(model_name=model_name)
        emb: BaseEmbeddingModel
        if isinstance(embedding_model, str):
            emb = SentenceTransformerEmbedding(model_name=embedding_model)
        else:
            emb = embedding_model

        vdb: BaseVectorStore
        if vector_store is not None:
            vdb = vector_store
        elif persist_dir:
            vdb = ChromaVectorStore(persist_directory=str(persist_dir), collection_name=collection_name)
        else:
            vdb = InMemoryVectorStore()

        return cls(
            chunker=chunker or RecursiveCharacterChunker(),
            embedding_model=emb,
            vector_store=vdb,
            llm_client=llm,
        )

    @classmethod
    def from_container(cls, container: Container) -> PolyRAG:
        """Construct a PolyRAG application context from an existing DI Container."""
        return cls(container=container)

    # -------------------------------------------------------------------------
    # Shared Document Ingestion
    # -------------------------------------------------------------------------

    def ingest_text(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk, embed, and index text into the shared vector store."""
        return self._default_pipeline.ingest_text(text=text, source=source, metadata=metadata)

    def ingest(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Convenience alias for ingest_text."""
        return self.ingest_text(text=text, source=source, metadata=metadata)

    def ingest_file(
        self,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Read and ingest a text or markdown file into the shared vector store."""
        return self._default_pipeline.ingest_file(file_path=file_path, metadata=metadata)

    def ingest_directory(
        self,
        dir_path: Path | str,
        glob_pattern: str = "*.txt",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recursively scan and ingest all matching files from a directory."""
        return self._default_pipeline.ingest_directory(
            dir_path=dir_path,
            glob_pattern=glob_pattern,
            metadata=metadata,
        )

    def ingest_documents(
        self,
        documents: Iterable[Any],
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ingest an iterable, generator, or list of documents.

        Supports:
        - LangChain Document objects (from .load() or .lazy_load())
        - PolyRAG Document models
        - Standard dictionaries ({"text": ..., "source": ...})
        - Raw strings
        """
        return self._default_pipeline.ingest_documents(documents=documents, metadata=metadata)

    def ingest_langchain_loader(
        self,
        loader: Any,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ingest documents from any LangChain DocumentLoader (e.g. PyPDFLoader, CSVLoader, WebBaseLoader).
        Streams memory-efficiently using loader.lazy_load() when available, falling back to loader.load().
        """
        return self._default_pipeline.ingest_langchain_loader(loader=loader, metadata=metadata)

    def ingest_langchain_documents(
        self,
        langchain_docs: Iterable[Any],
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Convenience alias for ingest_documents."""
        return self.ingest_documents(documents=langchain_docs, metadata=metadata)

    # -------------------------------------------------------------------------
    # Shared Direct Retrieval & Context Formatting
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Find the top-k most relevant chunks for a query from the shared store."""
        return self._default_pipeline.retrieve(query=query, top_k=top_k, **kwargs)

    def format_context(
        self,
        search_results: list[dict[str, Any]],
    ) -> str:
        """Format search results into a clean context block."""
        return self._default_pipeline.format_context(search_results)

    # -------------------------------------------------------------------------
    # Pipeline Factory Methods
    # -------------------------------------------------------------------------

    def create_naive_rag(self) -> NaiveRAG:
        """Create a NaiveRAG pipeline reusing the configured models and vector store."""
        return NaiveRAG(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            chunker=self.chunker,
        )

    def create_advanced_rag(
        self,
        top_k: int = 5,
        num_expanded_queries: int = 3,
        min_relevance_score: float = 0.0,
        verbose: bool = False,
    ) -> AdvancedRAG:
        """Create an AdvancedRAG pipeline with multi-query expansion and RRF fusion."""
        return AdvancedRAG(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            chunker=self.chunker,
            top_k=top_k,
            num_expanded_queries=num_expanded_queries,
            min_relevance_score=min_relevance_score,
            verbose=verbose,
        )

    def create_agentic_rag(
        self,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> AgenticRAG:
        """Create an AgenticRAG pipeline with planning, rewriting, and fused reflection."""
        return AgenticRAG(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )

    def create_agentic_service(
        self,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> Any:
        """Create an AgenticRAGService facade instance."""
        from polyrag.service import AgenticRAGService
        return AgenticRAGService(
            rag_service=self,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )

    def create_react_agent(
        self,
        max_steps: int = 4,
        default_top_k: int = 5,
        verbose: bool = False,
    ) -> ReActAgent:
        """Create a ReActAgent pipeline with Thought-Action-Observation loops."""
        return ReActAgent(
            llm_client=self.llm_client,
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            max_steps=max_steps,
            default_top_k=default_top_k,
            verbose=verbose,
        )

    # -------------------------------------------------------------------------
    # Execution Shortcut
    # -------------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """Shortcut: perform end-to-end retrieval and answering using default NaiveRAG."""
        return self._default_pipeline.execute(question=question, top_k=top_k)


__all__ = ["PolyRAG"]
