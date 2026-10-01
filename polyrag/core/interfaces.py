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

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        """
        Modern LangChain Runnable invoke interface.

        Accepts:
        - Plain string prompt
        - List of chat messages (BaseMessage instances, dicts, or tuples)

        Returns:
            AIMessage instance containing model output text.
        """
        from polyrag.core.models import AIMessage

        if isinstance(input, str):
            text = self.complete(input, **kwargs)
            return AIMessage(content=text)

        if isinstance(input, list):
            normalized: list[dict[str, str]] = []
            for m in input:
                if isinstance(m, tuple) and len(m) == 2:
                    role, content = m
                    if role in ("human", "user"):
                        r = "user"
                    elif role in ("ai", "assistant"):
                        r = "assistant"
                    else:
                        r = str(role)
                    normalized.append({"role": r, "content": str(content)})
                elif hasattr(m, "content"):
                    m_type = getattr(m, "type", "user")
                    r = "assistant" if m_type == "ai" else ("user" if m_type == "human" else str(m_type))
                    normalized.append({"role": r, "content": str(getattr(m, "content", ""))})
                elif isinstance(m, dict):
                    normalized.append({
                        "role": str(m.get("role", "user")),
                        "content": str(m.get("content", "")),
                    })
                else:
                    normalized.append({"role": "user", "content": str(m)})

            text = self.chat(normalized, **kwargs)
            return AIMessage(content=text)

        text = self.complete(str(input), **kwargs)
        return AIMessage(content=text)

    def stream(self, input: Any, **kwargs: Any) -> Any:
        """Stream chunks from the model if supported; yields complete response by default."""
        yield self.invoke(input, **kwargs)

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        """Bind tools to the model for tool-calling workflows."""
        return self

