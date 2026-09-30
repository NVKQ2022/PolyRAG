"""High-level developer-facing service facade for RAG and Agentic RAG."""

import os
from pathlib import Path
from typing import Any

from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import RAGResponse
from polyrag.embeddings.sentence_transformers import SentenceTransformerEmbedding
from polyrag.llms.openai import OpenAILLM
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.vector_stores.chroma import ChromaVectorStore
from polyrag.vector_stores.memory import InMemoryVectorStore


class RAGService:
    """
    High-level, production-ready facade for RAG workflows.

    Features:
    - Zero-configuration `.from_env()` or `.create()` initialization
    - Full backward compatibility with existing RAGService signatures
    - Ingest strings, files, or entire directories
    - Seamless upgrade to Agentic RAG via `.as_agentic()`
    """

    def __init__(
        self,
        client: Any = None,
        chunking_service: BaseChunker | None = None,
        embedding_service: BaseEmbeddingModel | None = None,
        vector_db: BaseVectorStore | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")

        # Resolve LLM adapter
        if client is not None and not isinstance(client, BaseLLMClient):
            self.llm_client: BaseLLMClient = OpenAILLM(client=client, model_name=self.model_name)
        elif isinstance(client, BaseLLMClient):
            self.llm_client = client
        else:
            self.llm_client = OpenAILLM(model_name=self.model_name)

        # Retain self.client for backward compatibility
        self.client = getattr(self.llm_client, "client", client)

        # Components
        self.chunking_service = chunking_service or RecursiveCharacterChunker()
        self.embedding_service = embedding_service or SentenceTransformerEmbedding()
        self.vector_db = vector_db or InMemoryVectorStore()

        # Internal pipeline
        self._pipeline = NaiveRAG(
            embedding_model=self.embedding_service,
            vector_store=self.vector_db,
            llm_client=self.llm_client,
            chunker=self.chunking_service,
        )

    @classmethod
    def create(
        cls,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str | Path | None = None,
        collection_name: str = "documents",
    ) -> "RAGService":
        """Factory method to construct a RAGService with sensible defaults."""
        llm = OpenAILLM(model_name=model_name)
        emb = SentenceTransformerEmbedding(model_name=embedding_model)
        vdb: BaseVectorStore
        if persist_dir:
            vdb = ChromaVectorStore(persist_path=persist_dir, collection_name=collection_name)
        else:
            vdb = InMemoryVectorStore()

        return cls(
            client=llm,
            chunking_service=RecursiveCharacterChunker(),
            embedding_service=emb,
            vector_db=vdb,
            model_name=model_name,
        )

    @classmethod
    def from_env(
        cls,
        persist_dir: str | Path | None = "./chroma_db",
        collection_name: str = "rfc_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> "RAGService":
        """
        Factory method reading configuration from environment variables (.env).
        Attempts to load persisted ChromaDB if present; falls back to InMemoryVectorStore.
        """
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        emb = SentenceTransformerEmbedding(model_name=embedding_model)

        vdb: BaseVectorStore
        try:
            if persist_dir and Path(persist_dir).exists():
                vdb = ChromaVectorStore(persist_path=persist_dir, collection_name=collection_name)
            else:
                vdb = InMemoryVectorStore()
        except Exception:
            vdb = InMemoryVectorStore()

        return cls(
            client=OpenAILLM(model_name=model_name),
            chunking_service=RecursiveCharacterChunker(),
            embedding_service=emb,
            vector_db=vdb,
            model_name=model_name,
        )

    @classmethod
    def from_container(cls, container: Any) -> "RAGService":
        """Instantiate RAGService using dependencies resolved from a DI Container."""
        from polyrag.core.interfaces import (
            BaseChunker,
            BaseEmbeddingModel,
            BaseLLMClient,
            BaseVectorStore,
        )

        llm = container.resolve(BaseLLMClient)
        chunker = container.resolve(BaseChunker) if container.is_registered(BaseChunker) else RecursiveCharacterChunker()
        emb = container.resolve(BaseEmbeddingModel)
        vdb = container.resolve(BaseVectorStore)

        return cls(
            client=llm,
            chunking_service=chunker,
            embedding_service=emb,
            vector_db=vdb,
            model_name=getattr(llm, "model_name", None),
        )

    def ingest(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk, embed, and store document text into the vector database."""
        return self._pipeline.ingest_text(text=text, source=source, metadata=metadata)

    def ingest_file(
        self,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Read and ingest a single document file."""
        return self._pipeline.ingest_file(file_path=file_path, metadata=metadata)

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
    ) -> list[dict[str, Any]]:
        """Find the top-k most relevant chunks for a query."""
        return self._pipeline.retrieve(query=query, top_k=top_k)

    def format_context(
        self,
        search_results: list[dict[str, Any]],
    ) -> str:
        """Format search results into a clean context string."""
        return self._pipeline.format_context(search_results)

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResponse:
        """Perform end-to-end RAG retrieval and answer generation."""
        return self._pipeline.execute(question=question, top_k=top_k)

    def as_agentic(
        self,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> "AgenticRAGService":
        """Convert this RAGService instance into an AgenticRAGService."""
        return AgenticRAGService(
            rag_service=self,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )


class AgenticRAGService:
    """
    Agentic RAG Service providing multi-round retrieval, query rewriting,
    deduplication, and fused reflection with source citations.
    """

    def __init__(
        self,
        rag_service: RAGService,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> None:
        self.rag_service = rag_service
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.verbose = verbose

        self._agentic_pipeline = AgenticRAG(
            embedding_model=rag_service.embedding_service,
            vector_store=rag_service.vector_db,
            llm_client=rag_service.llm_client,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )

    @classmethod
    def from_env(
        cls,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
        persist_dir: str | Path | None = "./chroma_db",
        collection_name: str = "rfc_documents",
    ) -> "AgenticRAGService":
        """Instantiate AgenticRAGService using environment configurations."""
        base_rag = RAGService.from_env(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        return cls(
            rag_service=base_rag,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )

    @classmethod
    def from_container(
        cls,
        container: Any,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> "AgenticRAGService":
        """Instantiate AgenticRAGService using dependencies resolved from a DI Container."""
        base_rag = RAGService.from_container(container)
        return cls(
            rag_service=base_rag,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )

    def decide_retrieval(self, question: str) -> dict[str, Any]:
        """Expose planning decision."""
        return self._agentic_pipeline.decide_retrieval(question)

    def direct_answer(self, question: str) -> str:
        """Expose direct conversational answer."""
        return self._agentic_pipeline.direct_answer(question)

    def reflect_and_evaluate(
        self,
        question: str,
        context: str,
        round_number: int,
    ) -> dict[str, Any]:
        """Expose reflection evaluation."""
        return self._agentic_pipeline.reflect_and_evaluate(
            question=question,
            context=context,
            round_number=round_number,
            max_rounds=self.max_rounds,
        )

    def query(self, question: str) -> RAGResponse:
        """Execute agentic multi-round query answering."""
        return self._agentic_pipeline.execute(
            question=question,
            top_k=self.top_k,
            max_rounds=self.max_rounds,
        )


__all__ = [
    "RAGService",
    "AgenticRAGService",
]
