"""Abstract Base Classes and Interfaces (Ports) for polyrag with native LangChain bridges."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter


class BaseChunker(TextSplitter, ABC):
    """
    Abstract interface for document chunking strategies.
    Inherits from LangChain TextSplitter while maintaining PolyRAG chunk() port.
    """

    def __init__(
        self,
        chunk_size: int = 550,
        chunk_overlap: int = 35,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """LangChain standard method: split text into string chunks."""
        return self.chunk(text)

    def chunk(self, text: str) -> list[str]:
        """PolyRAG port: split input document text into discrete chunks."""
        return self.split_text(text)


class BaseEmbeddingModel(Embeddings, ABC):
    """
    Abstract interface for text embedding generation.
    Inherits from LangChain Embeddings while maintaining PolyRAG ports.
    """

    @property
    def dim(self) -> int:
        """Return dimensionality of the embedding vector."""
        return 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """LangChain standard: embed list of texts."""
        if hasattr(self, "embed_batch"):
            return self.embed_batch(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """LangChain standard: embed single query text."""
        if hasattr(self, "embed_text"):
            return self.embed_text(text)
        docs = self.embed_documents([text])
        return docs[0] if docs else []

    def embed_text(self, text: str) -> list[float]:
        """PolyRAG port: generate embedding vector for a single text."""
        return self.embed_query(text)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 128,
    ) -> list[list[float]]:
        """PolyRAG port: generate normalized embeddings for multiple texts."""
        return self.embed_documents(texts)


class BaseVectorStore(VectorStore, ABC):
    """
    Abstract interface for vector database storage and retrieval.
    Inherits from LangChain VectorStore while maintaining PolyRAG ports.
    """

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """LangChain standard constructor from raw texts."""
        store = cls(**kwargs)
        if hasattr(store, "add_texts"):
            store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[LCDocument]:
        """LangChain standard similarity search."""
        if hasattr(self, "search"):
            emb = getattr(self, "embedding_model", None) or getattr(self, "embedding", None)
            if emb:
                vec = emb.embed_query(query) if hasattr(emb, "embed_query") else emb.embed_text(query)
                results = self.search(query_vector=vec, top_k=k, **kwargs)
                return [
                    LCDocument(
                        page_content=r.get("document", {}).get("text", r.get("document", {}).get("page_content", "")),
                        metadata=r.get("document", {}),
                    )
                    for r in results
                ]
        return []

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
    """
    Abstract interface for interacting with Language Models and ChatModels.
    Conforms to modern LangChain Runnable invocation schemas.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return active model name."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion for a prompt."""
        raise NotImplementedError

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate and parse structured JSON from a prompt."""
        import json
        import re

        raw_text = self.complete(prompt, **kwargs)
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        candidate = match.group(1) if match else raw_text
        candidate = candidate.strip()

        first_brace = candidate.find("{")
        last_brace = candidate.rfind("}")
        if first_brace != -1 and last_brace != -1:
            candidate = candidate[first_brace : last_brace + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                import ast
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Generate chat response for conversation messages."""
        prompt = "\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages)
        return self.complete(prompt, **kwargs)

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        """
        Modern LangChain Runnable invoke interface.
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
                    r = "assistant" if role in ("ai", "assistant") else ("user" if role in ("human", "user") else str(role))
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
                    normalized.append({"role": "user", "content": str(input)})

            text = self.chat(normalized, **kwargs)
            return AIMessage(content=text)

        text = self.complete(str(input), **kwargs)
        return AIMessage(content=text)

    def stream(self, input: Any, **kwargs: Any) -> Any:
        """Stream chunks from the model if supported."""
        yield self.invoke(input, **kwargs)

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        """Bind tools to the model for tool-calling workflows."""
        return self


class BaseRAGInterface(ABC):
    """Abstract interface for RAG pipelines."""

    @abstractmethod
    def execute(self, question: str, **kwargs: Any) -> Any:
        raise NotImplementedError


__all__ = [
    "BaseVectorStore",
    "BaseEmbeddingModel",
    "BaseLLMClient",
    "BaseChunker",
    "BaseRAGInterface",
    "VectorStore",
    "Embeddings",
    "BaseChatModel",
    "TextSplitter",
]
