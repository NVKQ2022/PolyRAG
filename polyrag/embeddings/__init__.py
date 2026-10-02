"""Embeddings Package for polyrag built on langchain_core.embeddings."""

from typing import Any

from langchain_core.embeddings import Embeddings, FakeEmbeddings

from polyrag.core.interfaces import BaseEmbeddingModel
from polyrag.embeddings.openai import OpenAIEmbedding
from polyrag.embeddings.sentence_transformers import SentenceTransformerEmbedding


def resolve_embedding_model(
    embedding_model: Embeddings | str | Any | None = None,
    **kwargs: Any,
) -> Embeddings:
    """
    Resolve an embedding strategy into a concrete LangChain Embeddings instance.

    Supported input types:
    1. Embeddings instance: returned directly.
    2. None: defaults to SentenceTransformerEmbedding("all-MiniLM-L6-v2") or kwargs.
    3. String strategy identifier / model name:
       - "openai": instantiates OpenAIEmbedding(**kwargs)
       - "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002",
         or any string starting with "text-embedding-":
         instantiates OpenAIEmbedding(model_name=embedding_model, **kwargs)
       - "fake": instantiates FakeEmbeddings(size=kwargs.get("size", 384))
       - "sentence-transformers", "sentence_transformers", "st", "local":
         instantiates SentenceTransformerEmbedding(**kwargs)
       - Any other model name (e.g. "all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "all-mpnet-base-v2"):
         instantiates SentenceTransformerEmbedding(model_name=embedding_model, **kwargs)
    4. LangChain Embeddings duck-typed instance: returned directly or adapted.

    Args:
        embedding_model: Embedding model instance, string identifier, or LangChain Embeddings.
        **kwargs: Optional constructor arguments passed when instantiating new models.

    Returns:
        Embeddings instance.
    """
    if embedding_model is None:
        return SentenceTransformerEmbedding(**kwargs)

    if isinstance(embedding_model, Embeddings):
        return embedding_model

    if isinstance(embedding_model, str):
        cleaned = embedding_model.strip()
        lower = cleaned.lower()

        if lower == "fake":
            size = kwargs.get("size", 384)
            return FakeEmbeddings(size=size)

        if lower in ("sentence-transformers", "sentence_transformers", "st", "local"):
            return SentenceTransformerEmbedding(**kwargs)

        if lower == "openai":
            return OpenAIEmbedding(**kwargs)

        if lower in ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002") or lower.startswith("text-embedding-"):
            return OpenAIEmbedding(model_name=cleaned, **kwargs)

        # Default string fallback treats it as a HuggingFace / SentenceTransformers model
        return SentenceTransformerEmbedding(model_name=cleaned, **kwargs)

    # Duck-typing check for custom LangChain embeddings
    if hasattr(embedding_model, "embed_documents") and hasattr(embedding_model, "embed_query"):
        from polyrag.adapters.langchain import LangChainEmbeddingAdapter
        return LangChainEmbeddingAdapter(embedding_model)

    raise TypeError(
        f"Expected BaseEmbeddingModel instance or string model name, got {type(embedding_model).__name__}"
    )


__all__ = [
    "BaseEmbeddingModel",
    "Embeddings",
    "FakeEmbeddings",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    "resolve_embedding_model",
]
