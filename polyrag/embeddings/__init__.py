"""Embeddings Package for polyrag."""

from typing import Any

from polyrag.core.interfaces import BaseEmbeddingModel
from polyrag.embeddings.openai import OpenAIEmbedding
from polyrag.embeddings.sentence_transformers import SentenceTransformerEmbedding


def resolve_embedding_model(
    embedding_model: BaseEmbeddingModel | str | Any | None = None,
    **kwargs: Any,
) -> BaseEmbeddingModel:
    """
    Resolve an embedding strategy into a concrete BaseEmbeddingModel instance.

    Supported input types:
    1. BaseEmbeddingModel instance: returned directly.
    2. None: defaults to SentenceTransformerEmbedding("all-MiniLM-L6-v2") or kwargs.
    3. String strategy identifier / model name:
       - "openai": instantiates OpenAIEmbedding(**kwargs)
       - "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002",
         or any string starting with "text-embedding-":
         instantiates OpenAIEmbedding(model_name=embedding_model, **kwargs)
       - "sentence-transformers", "sentence_transformers", "st", "local":
         instantiates SentenceTransformerEmbedding(**kwargs)
       - Any other model name (e.g. "all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "all-mpnet-base-v2"):
         instantiates SentenceTransformerEmbedding(model_name=embedding_model, **kwargs)
    4. LangChain Embeddings duck-typed instance (implements 'embed_documents' and 'embed_query'):
       wrapped into a LangChainEmbeddingAdapter.

    Args:
        embedding_model: Embedding model instance, string identifier, or LangChain Embeddings.
        **kwargs: Optional constructor arguments passed when instantiating new models.

    Returns:
        BaseEmbeddingModel instance.
    """
    if embedding_model is None:
        return SentenceTransformerEmbedding(**kwargs)

    if isinstance(embedding_model, BaseEmbeddingModel):
        return embedding_model

    if isinstance(embedding_model, str):
        cleaned = embedding_model.strip()
        lower = cleaned.lower()

        if lower in ("sentence-transformers", "sentence_transformers", "st", "local"):
            return SentenceTransformerEmbedding(**kwargs)

        if lower == "openai":
            return OpenAIEmbedding(**kwargs)

        if lower in ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002") or lower.startswith("text-embedding-"):
            return OpenAIEmbedding(model_name=cleaned, **kwargs)

        # Default string fallback treats it as a HuggingFace / SentenceTransformers model
        return SentenceTransformerEmbedding(model_name=cleaned, **kwargs)

    # Check duck typing for LangChain Embeddings interface
    if hasattr(embedding_model, "embed_documents") and hasattr(embedding_model, "embed_query"):
        from polyrag.adapters.langchain import LangChainEmbeddingAdapter
        return LangChainEmbeddingAdapter(embedding_model, **kwargs)

    raise TypeError(
        f"Expected BaseEmbeddingModel instance, string model name, or LangChain Embeddings object, got {type(embedding_model).__name__}"
    )


__all__ = [
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    "resolve_embedding_model",
]
