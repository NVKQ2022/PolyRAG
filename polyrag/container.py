"""Dependency Injection (DI) Container for PolyRAG."""

from collections.abc import Callable
import os
from pathlib import Path
from typing import Any, TypeVar

from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.embeddings.sentence_transformers import SentenceTransformerEmbedding
from polyrag.exceptions import ConfigurationError
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

    # =========================================================================
    # Factory Constructors
    # =========================================================================

    @classmethod
    def create(
        cls,
        chunker: BaseChunker | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        vector_store: BaseVectorStore | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> "Container":
        """Initialize container with optional custom dependencies or sensible defaults."""
        container = cls()

        # Defaults
        chunker_instance = chunker or RecursiveCharacterChunker()
        embedding_instance = embedding_model or SentenceTransformerEmbedding()
        vector_store_instance = vector_store or InMemoryVectorStore()
        llm_instance = llm_client or OpenAILLM()

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
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> "Container":
        """Initialize container resolving configuration from environment variables."""
        container = cls()
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

        # Chunker
        container.register_instance(BaseChunker, RecursiveCharacterChunker())

        # Embedding
        container.register_instance(
            BaseEmbeddingModel,
            SentenceTransformerEmbedding(model_name=embedding_model),
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

    def build_naive_rag(self) -> NaiveRAG:
        """Construct NaiveRAG pipeline using injected dependencies."""
        return NaiveRAG(
            embedding_model=self.resolve(BaseEmbeddingModel),
            vector_store=self.resolve(BaseVectorStore),
            llm_client=self.resolve(BaseLLMClient),
            chunker=self.resolve(BaseChunker),
        )

    def build_agentic_rag(
        self,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> AgenticRAG:
        """Construct AgenticRAG pipeline using injected dependencies."""
        return AgenticRAG(
            embedding_model=self.resolve(BaseEmbeddingModel),
            vector_store=self.resolve(BaseVectorStore),
            llm_client=self.resolve(BaseLLMClient),
            top_k=top_k,
            max_rounds=max_rounds,
            verbose=verbose,
        )

    def build_react_agent(
        self,
        max_steps: int = 4,
        default_top_k: int = 5,
        verbose: bool = False,
    ) -> ReActAgent:
        """Construct ReActAgent pipeline using injected dependencies."""
        return ReActAgent(
            llm_client=self.resolve(BaseLLMClient),
            embedding_model=self.resolve(BaseEmbeddingModel),
            vector_store=self.resolve(BaseVectorStore),
            max_steps=max_steps,
            default_top_k=default_top_k,
            verbose=verbose,
        )

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
