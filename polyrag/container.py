"""Dependency Injection (DI) Container for PolyRAG."""

from collections.abc import Callable
import os
from pathlib import Path
from typing import Any, TypeVar

from polyrag.chunkers import resolve_chunker
from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.embeddings import resolve_embedding_model
from polyrag.embeddings.sentence_transformers import SentenceTransformerEmbedding
from polyrag.exceptions import ConfigurationError
from polyrag.llms import resolve_llm_client
from polyrag.llms.openai import OpenAILLM
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent
from polyrag.vector_stores.chroma import ChromaVectorStore
from polyrag.vector_stores.memory import InMemoryVectorStore

T = TypeVar("T")


class Container:
    """
    Lightweight, thread-safe Dependency Injection (DI) container.

    Acts as the Composition Root for resolving, configuring, and wiring
    all PolyRAG components and pipelines.
    """

    def __init__(self) -> None:
        self._singletons: dict[type, Any] = {}
        self._factories: dict[type, Callable[..., Any]] = {}

    def register_instance(self, interface: type[T], instance: T) -> "Container":
        """Register a pre-existing concrete instance as a singleton for an interface."""
        self._singletons[interface] = instance
        # Clear factory if previously set
        self._factories.pop(interface, None)
        return self

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[..., T],
        singleton: bool = True,
    ) -> "Container":
        """Register a factory callable to construct instances of an interface."""
        if singleton:
            # Lazy singleton wrapper
            def singleton_factory() -> Any:
                if interface not in self._singletons:
                    self._singletons[interface] = factory(self) if _accepts_arg(factory) else factory()
                return self._singletons[interface]

            self._factories[interface] = singleton_factory
        else:
            self._factories[interface] = factory
        return self

    def resolve(self, interface: type[T]) -> T:
        """Resolve an implementation for the requested interface."""
        # 1. Direct singleton instance
        if interface in self._singletons:
            return self._singletons[interface]

        # 2. Registered factory
        if interface in self._factories:
            factory = self._factories[interface]
            return factory(self) if _accepts_arg(factory) else factory()

        raise ConfigurationError(
            f"No implementation registered for interface '{interface.__name__}' in Container."
        )

    def is_registered(self, interface: type) -> bool:
        """Check if an interface has a registered instance or factory."""
        return interface in self._singletons or interface in self._factories

    @property
    def chunker(self) -> BaseChunker:
        """Resolve registered chunker."""
        return self.resolve(BaseChunker)

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        """Resolve registered embedding model."""
        return self.resolve(BaseEmbeddingModel)

    @property
    def vector_store(self) -> BaseVectorStore:
        """Resolve registered vector store."""
        return self.resolve(BaseVectorStore)

    @property
    def llm_client(self) -> BaseLLMClient:
        """Resolve registered LLM client."""
        return self.resolve(BaseLLMClient)

    @property
    def chat_model(self) -> BaseLLMClient:
        """Alias for llm_client resolving registered ChatModel / LLM client."""
        return self.resolve(BaseLLMClient)

    # =========================================================================
    # Factory Constructors
    # =========================================================================

    @classmethod
    def create(
        cls,
        chunker: BaseChunker | str | None = None,
        embedding_model: BaseEmbeddingModel | str | Any | None = None,
        vector_store: BaseVectorStore | None = None,
        llm_client: BaseLLMClient | str | Any | None = None,
        embedding: BaseEmbeddingModel | str | Any | None = None,
        chat_model: BaseLLMClient | str | Any | None = None,
        llm: BaseLLMClient | str | Any | None = None,
    ) -> "Container":
        """Initialize container with optional custom dependencies or sensible defaults."""
        container = cls()

        target_llm = chat_model if chat_model is not None else (llm if llm is not None else llm_client)
        # Defaults
        chunker_instance = resolve_chunker(chunker)
        embedding_instance = resolve_embedding_model(embedding if embedding is not None else embedding_model)
        vector_store_instance = vector_store or InMemoryVectorStore()
        llm_instance = resolve_llm_client(target_llm)

        container.register_instance(BaseChunker, chunker_instance)
        container.register_instance(BaseEmbeddingModel, embedding_instance)
        container.register_instance(BaseVectorStore, vector_store_instance)
        container.register_instance(BaseLLMClient, llm_instance)

        return container

    @classmethod
    def from_env(
        cls,
        persist_dir: str | Path | None = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: BaseEmbeddingModel | str | Any | None = "all-MiniLM-L6-v2",
        embedding: BaseEmbeddingModel | str | Any | None = None,
    ) -> "Container":
        """Initialize container resolving configuration from environment variables."""
        container = cls()
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

        # Chunker
        container.register_instance(BaseChunker, RecursiveCharacterChunker())

        # Embedding
        container.register_instance(
            BaseEmbeddingModel,
            resolve_embedding_model(embedding if embedding is not None else embedding_model),
        )

        # Vector Store
        try:
            if persist_dir and Path(persist_dir).exists():
                vdb: BaseVectorStore = ChromaVectorStore(
                    persist_path=persist_dir,
                    collection_name=collection_name,
                )
            else:
                vdb = InMemoryVectorStore()
        except Exception:
            vdb = InMemoryVectorStore()
        container.register_instance(BaseVectorStore, vdb)

        # LLM Client
        container.register_instance(BaseLLMClient, OpenAILLM(model_name=model_name))

        return container

    # =========================================================================
    # Pipeline Builders
    # =========================================================================

    def build_naive_rag(
        self,
        chunker: BaseChunker | str | None = None,
        embedding_model: BaseEmbeddingModel | str | Any | None = None,
        embedding: BaseEmbeddingModel | str | Any | None = None,
        llm_client: BaseLLMClient | str | Any | None = None,
        chat_model: BaseLLMClient | str | Any | None = None,
        llm: BaseLLMClient | str | Any | None = None,
    ) -> NaiveRAG:
        """Construct NaiveRAG pipeline using injected dependencies."""
        resolved_chunker = (
            resolve_chunker(chunker)
            if chunker is not None
            else (self.resolve(BaseChunker) if self.is_registered(BaseChunker) else None)
        )
        target_emb = embedding if embedding is not None else embedding_model
        resolved_embedding = (
            resolve_embedding_model(target_emb)
            if target_emb is not None
            else self.resolve(BaseEmbeddingModel)
        )
        target_llm = chat_model if chat_model is not None else (llm if llm is not None else llm_client)
        resolved_llm = (
            resolve_llm_client(target_llm)
            if target_llm is not None
            else self.resolve(BaseLLMClient)
        )
        return NaiveRAG(
            embedding_model=resolved_embedding,
            vector_store=self.resolve(BaseVectorStore),
            llm_client=resolved_llm,
            chunker=resolved_chunker,
        )

    def build_advanced_rag(
        self,
        chunker: BaseChunker | str | None = None,
        embedding_model: BaseEmbeddingModel | str | Any | None = None,
        top_k: int = 5,
        num_expanded_queries: int = 3,
        min_relevance_score: float = 0.0,
        verbose: bool = False,
        embedding: BaseEmbeddingModel | str | Any | None = None,
        llm_client: BaseLLMClient | str | Any | None = None,
        chat_model: BaseLLMClient | str | Any | None = None,
        llm: BaseLLMClient | str | Any | None = None,
    ) -> Any:
        """Construct AdvancedRAG pipeline using injected dependencies."""
        from polyrag.pipelines.advanced import AdvancedRAG
        resolved_chunker = (
            resolve_chunker(chunker)
            if chunker is not None
            else (self.resolve(BaseChunker) if self.is_registered(BaseChunker) else None)
        )
        target_emb = embedding if embedding is not None else embedding_model
        resolved_embedding = (
            resolve_embedding_model(target_emb)
            if target_emb is not None
            else self.resolve(BaseEmbeddingModel)
        )
        target_llm = chat_model if chat_model is not None else (llm if llm is not None else llm_client)
        resolved_llm = (
            resolve_llm_client(target_llm)
            if target_llm is not None
            else self.resolve(BaseLLMClient)
        )
        return AdvancedRAG(
            embedding_model=resolved_embedding,
            vector_store=self.resolve(BaseVectorStore),
            llm_client=resolved_llm,
            chunker=resolved_chunker,
            top_k=top_k,
            num_expanded_queries=num_expanded_queries,
            min_relevance_score=min_relevance_score,
            verbose=verbose,
        )

    def build_agentic_rag(
        self,
        chunker: BaseChunker | str | None = None,
        embedding_model: BaseEmbeddingModel | str | Any | None = None,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
        embedding: BaseEmbeddingModel | str | Any | None = None,
        llm_client: BaseLLMClient | str | Any | None = None,
        chat_model: BaseLLMClient | str | Any | None = None,
        llm: BaseLLMClient | str | Any | None = None,
        tools: list[Any] | None = None,
        state_schema: type | dict | None = None,
        system_prompt: str | None = None,
    ) -> AgenticRAG:
        """Construct AgenticRAG pipeline using injected dependencies."""
        resolved_chunker = (
            resolve_chunker(chunker)
            if chunker is not None
            else (self.resolve(BaseChunker) if self.is_registered(BaseChunker) else None)
        )
        target_emb = embedding if embedding is not None else embedding_model
        resolved_embedding = (
            resolve_embedding_model(target_emb)
            if target_emb is not None
            else self.resolve(BaseEmbeddingModel)
        )
        target_llm = chat_model if chat_model is not None else (llm if llm is not None else llm_client)
        resolved_llm = (
            resolve_llm_client(target_llm)
            if target_llm is not None
            else self.resolve(BaseLLMClient)
        )
        return AgenticRAG(
            embedding_model=resolved_embedding,
            vector_store=self.resolve(BaseVectorStore),
            llm_client=resolved_llm,
            chunker=resolved_chunker,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
            tools=tools,
            state_schema=state_schema,
            system_prompt=system_prompt,
        )

    def build_react_agent(
        self,
        chunker: BaseChunker | str | None = None,
        embedding_model: BaseEmbeddingModel | str | Any | None = None,
        max_steps: int = 4,
        default_top_k: int = 5,
        verbose: bool = False,
        embedding: BaseEmbeddingModel | str | Any | None = None,
        llm_client: BaseLLMClient | str | Any | None = None,
        chat_model: BaseLLMClient | str | Any | None = None,
        llm: BaseLLMClient | str | Any | None = None,
    ) -> ReActAgent:
        """Construct ReActAgent pipeline using injected dependencies."""
        resolved_chunker = (
            resolve_chunker(chunker)
            if chunker is not None
            else (self.resolve(BaseChunker) if self.is_registered(BaseChunker) else None)
        )
        target_emb = embedding if embedding is not None else embedding_model
        resolved_embedding = (
            resolve_embedding_model(target_emb)
            if target_emb is not None
            else self.resolve(BaseEmbeddingModel)
        )
        target_llm = chat_model if chat_model is not None else (llm if llm is not None else llm_client)
        resolved_llm = (
            resolve_llm_client(target_llm)
            if target_llm is not None
            else self.resolve(BaseLLMClient)
        )
        return ReActAgent(
            llm_client=resolved_llm,
            embedding_model=resolved_embedding,
            vector_store=self.resolve(BaseVectorStore),
            chunker=resolved_chunker,
            max_steps=max_steps,
            default_top_k=default_top_k,
            verbose=verbose,
        )

    def build_app(self) -> Any:
        """Construct PolyRAG application context using injected dependencies."""
        from polyrag.app import PolyRAG
        return PolyRAG.from_container(self)

    def build_service(self) -> Any:
        """Construct RAGService facade using injected dependencies."""
        from polyrag.service import RAGService
        return RAGService.from_container(self)

    def build_agentic_service(
        self,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> Any:
        """Construct AgenticRAGService facade using injected dependencies."""
        from polyrag.service import AgenticRAGService
        return AgenticRAGService.from_container(
            self,
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )


def _accepts_arg(fn: Callable[..., Any]) -> bool:
    """Check if callable accepts at least one positional argument."""
    import inspect
    try:
        sig = inspect.signature(fn)
        return len(sig.parameters) > 0
    except Exception:
        return False


__all__ = ["Container"]
