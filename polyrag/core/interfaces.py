"""Abstract Base Classes and Interfaces (Ports) for polyrag."""

from abc import ABC, abstractmethod
from typing import Any


class BaseChunker(ABC):
    """Abstract interface for document chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split input document text into discrete chunks."""
        raise NotImplementedError


class BaseEmbeddingModel(ABC):
    """Abstract interface for text embedding generation."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return dimensionality of the embedding vector."""
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 128,
    ) -> list[list[float]]:
        """Generate normalized embeddings for multiple texts."""
        raise NotImplementedError


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage and retrieval."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all records from the vector store."""
        raise NotImplementedError

    @abstractmethod
    def add_documents(
        self,
        vectors: list[list[float]],
        documents: list[dict[str, Any]],
        batch_size: int = 5000,
        **kwargs: Any,
    ) -> None:
        """Add pre-computed vectors and document records."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Perform nearest-neighbor search for a query embedding vector."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return total document count in the vector collection."""
        raise NotImplementedError

    @abstractmethod
    def peek(self, limit: int = 5) -> Any:
        """Preview sample records from the vector store."""
        raise NotImplementedError


class BaseLLMClient(ABC):
    """Abstract interface for interacting with Language Models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return active model name."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion for a prompt."""
        raise NotImplementedError

    @abstractmethod
    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate and parse structured JSON from a prompt."""
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Generate chat response for conversation messages."""
        raise NotImplementedError
