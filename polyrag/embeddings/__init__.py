"""Embeddings Package for PolyRAG built on langchain_core.embeddings."""

from typing import Any
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from polyrag.core.interfaces import BaseEmbeddingModel


class DuckTypedEmbeddingWrapper(BaseEmbeddingModel):
    """Simple wrapper ensuring duck-typed embedding objects satisfy Embeddings protocol."""

    def __init__(self, target: Any) -> None:
        self.target = target

    @property
    def dim(self) -> int:
        if hasattr(self.target, "dim"):
            return int(self.target.dim)
        if hasattr(self.target, "dimension"):
            return int(self.target.dimension)
        sample = self.embed_query("probe")
        return len(sample)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.target, "embed_documents"):
            return self.target.embed_documents(texts)
        if hasattr(self.target, "embed_batch"):
            return self.target.embed_batch(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if hasattr(self.target, "embed_query"):
            return self.target.embed_query(text)
        if hasattr(self.target, "embed_text"):
            return self.target.embed_text(text)
        return [0.0] * 384


def resolve_embedding_model(
    embedding_model: Embeddings | str | Any | None = None,
    **kwargs: Any,
) -> Embeddings:
    """
    Resolve an embedding strategy into a concrete LangChain Embeddings instance.

    Supported input types:
    1. Any native LangChain Embeddings instance: returned directly.
    2. None: defaults to FakeEmbeddings(size=384) for zero-dependency local use.
    3. String model identifier:
       - "fake": instantiates FakeEmbeddings(size=kwargs.get("size", 384))
       - "openai" or "text-embedding-*": instantiates langchain_openai.OpenAIEmbeddings
       - Any other model identifier: instantiates langchain_huggingface.HuggingFaceEmbeddings
    4. Duck-typed object implementing embed_documents/embed_query or embed_batch/embed_text.

    Args:
        embedding_model: Embedding model instance, string identifier, or LangChain Embeddings.
        **kwargs: Optional constructor arguments passed when instantiating new models.

    Returns:
        Embeddings instance.
    """
    if embedding_model is None:
        return FakeEmbeddings(size=kwargs.get("size", 384))

    if isinstance(embedding_model, Embeddings):
        return embedding_model

    if isinstance(embedding_model, str):
        cleaned = embedding_model.strip()
        lower = cleaned.lower()

        if lower == "fake":
            return FakeEmbeddings(size=kwargs.get("size", 384))

        if lower == "openai" or lower.startswith("text-embedding-"):
            try:
                from langchain_openai import OpenAIEmbeddings
                model_arg = {} if lower == "openai" else {"model": cleaned}
                return OpenAIEmbeddings(**model_arg, **kwargs)
            except ImportError as e:
                raise ImportError(
                    "To use OpenAI embeddings via string alias, install langchain-openai: "
                    "pip install langchain-openai"
                ) from e

        # Fallback to HuggingFace embeddings via official LangChain package
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=cleaned, **kwargs)
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=cleaned, **kwargs)
            except ImportError:
                # If neither is installed, return lightweight FakeEmbeddings for zero-setup dev
                return FakeEmbeddings(size=kwargs.get("size", 384))

    if (hasattr(embedding_model, "embed_documents") and hasattr(embedding_model, "embed_query")) or (
        hasattr(embedding_model, "embed_batch") and hasattr(embedding_model, "embed_text")
    ):
        return DuckTypedEmbeddingWrapper(embedding_model)

    raise TypeError(
        f"Expected BaseEmbeddingModel instance or string model name, got {type(embedding_model).__name__}"
    )


# Backward compatibility alias
LangChainEmbeddingAdapter = DuckTypedEmbeddingWrapper

__all__ = [
    "BaseEmbeddingModel",
    "Embeddings",
    "FakeEmbeddings",
    "DuckTypedEmbeddingWrapper",
    "LangChainEmbeddingAdapter",
    "resolve_embedding_model",
]
