"""High-level developer-facing service facade for RAG and Agentic RAG."""

import os
from pathlib import Path
from typing import Any

from polyrag.chunkers import resolve_chunker
from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import RAGResponse
from polyrag.embeddings import resolve_embedding_model
from polyrag.embeddings.sentence_transformers import SentenceTransformerEmbedding
from polyrag.llms.openai import OpenAILLM
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.vector_stores.chroma import ChromaVectorStore
from polyrag.vector_stores.memory import InMemoryVectorStore


from polyrag.app import PolyRAG


class RAGService(PolyRAG):
    """
    High-level, production-ready facade for RAG workflows.
    Retained for full backward compatibility; for new projects, prefer `PolyRAG`.

    Features:
    - Zero-configuration `.from_env()` or `.create()` initialization
    - Full backward compatibility with existing RAGService signatures
    - Ingest strings, files, or entire directories
    - Seamless creation of Advanced and Agentic RAG via `.create_advanced_rag()` and `.create_agentic_rag()`
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
            resolved_llm: BaseLLMClient = OpenAILLM(client=client, model_name=self.model_name)
        elif isinstance(client, BaseLLMClient):
            resolved_llm = client
        else:
            resolved_llm = OpenAILLM(model_name=self.model_name)

        super().__init__(
            chunker=chunking_service,
            embedding_model=embedding_service,
            vector_store=vector_db,
            llm_client=resolved_llm,
        )

        # Backward compatibility attribute aliases
        self.client = getattr(self.llm_client, "client", client)
        self.chunking_service = self.chunker
        self.embedding_service = self.embedding_model
        self.vector_db = self.vector_store
        self._pipeline = self._default_pipeline

    @classmethod
    def create(
        cls,
        model_name: str = "gpt-4o-mini",
        embedding_model: BaseEmbeddingModel | str | Any | None = "all-MiniLM-L6-v2",
        persist_dir: str | Path | None = None,
        collection_name: str = "documents",
        chunking_service: BaseChunker | str | None = None,
        embedding: BaseEmbeddingModel | str | Any | None = None,
        vector_db: BaseVectorStore | None = None,
    ) -> "RAGService":
        """Factory method to construct a RAGService with sensible defaults."""
        llm = OpenAILLM(model_name=model_name)
        target_emb = embedding if embedding is not None else embedding_model
        emb = resolve_embedding_model(target_emb)
        vdb: BaseVectorStore
        if vector_db is not None:
            vdb = vector_db
        elif persist_dir:
            vdb = ChromaVectorStore(persist_path=persist_dir, collection_name=collection_name)
        else:
            vdb = InMemoryVectorStore()

        return cls(
            client=llm,
            chunking_service=resolve_chunker(chunking_service),
            embedding_service=emb,
            vector_db=vdb,
            model_name=model_name,
        )

    @classmethod
    def from_env(
        cls,
        persist_dir: str | Path | None = "./chroma_db",
        collection_name: str = "rfc_documents",
        embedding_model: BaseEmbeddingModel | str | Any | None = "all-MiniLM-L6-v2",
        embedding: BaseEmbeddingModel | str | Any | None = None,
    ) -> "RAGService":
        """
        Factory method reading configuration from environment variables (.env).
        Attempts to load persisted ChromaDB if present; falls back to InMemoryVectorStore.
        """
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        target_emb = embedding if embedding is not None else embedding_model
        emb = resolve_embedding_model(target_emb)

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

    def create_agentic_rag(
        self,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> "AgenticRAGService":
        """Create an AgenticRAGService instance reusing this service's components."""
        return AgenticRAGService(
            rag_service=self,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )

    # Backward compatibility aliases
    def as_advanced(
        self,
        top_k: int = 5,
        num_expanded_queries: int = 3,
        min_relevance_score: float = 0.0,
        verbose: bool = False,
    ) -> Any:
        """Alias for create_advanced_rag (retained for backward compatibility)."""
        return self.create_advanced_rag(
            top_k=top_k,
            num_expanded_queries=num_expanded_queries,
            min_relevance_score=min_relevance_score,
            verbose=verbose,
        )

    def as_agentic(
        self,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> "AgenticRAGService":
        """Alias for create_agentic_rag (retained for backward compatibility)."""
        return self.create_agentic_rag(
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
