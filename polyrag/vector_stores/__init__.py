"""Vector Stores Package for PolyRAG built on langchain_core.vectorstores."""

from typing import Any
from langchain_core.vectorstores import VectorStore

from polyrag.core.interfaces import BaseVectorStore
from polyrag.vector_stores.memory import InMemoryVectorStore


def resolve_vector_store(
    vector_store: VectorStore | str | None = None,
    embedding: Any | None = None,
    **kwargs: Any,
) -> VectorStore:
    """
    Resolve or construct a LangChain-compatible VectorStore.

    Supports:
    - Any native LangChain VectorStore instance: returned directly.
    - None or "memory" / "in_memory": instantiates PolyRAG's InMemoryVectorStore.
    - "chroma": instantiates langchain_chroma.Chroma.
    - "milvus": instantiates langchain_milvus.Milvus.

    Args:
        vector_store: VectorStore instance or string name ('memory', 'chroma', 'milvus').
        embedding: Optional embedding model to attach to newly created stores.
        **kwargs: Additional keyword arguments forwarded to the vector store constructor.

    Returns:
        VectorStore instance.
    """
    if vector_store is None:
        return InMemoryVectorStore(embedding=embedding, **kwargs)

    if isinstance(vector_store, VectorStore):
        return vector_store

    if isinstance(vector_store, str):
        normalized = vector_store.lower().strip()
        if normalized in ("memory", "in_memory", "inmemory"):
            return InMemoryVectorStore(embedding=embedding, **kwargs)

        if normalized == "chroma":
            try:
                from langchain_chroma import Chroma
                return Chroma(embedding_function=embedding, **kwargs)
            except ImportError as e:
                raise ImportError(
                    "To use Chroma vector store via string alias, install langchain-chroma: "
                    "pip install langchain-chroma"
                ) from e

        if normalized in ("milvus", "milvus_lite", "lite"):
            try:
                from langchain_milvus import Milvus
                return Milvus(embedding_function=embedding, **kwargs)
            except ImportError as e:
                raise ImportError(
                    "To use Milvus vector store via string alias, install langchain-milvus: "
                    "pip install langchain-milvus"
                ) from e

    raise TypeError(
        f"Expected VectorStore instance or recognized store name, got {type(vector_store).__name__}"
    )


__all__ = [
    "BaseVectorStore",
    "VectorStore",
    "InMemoryVectorStore",
    "resolve_vector_store",
]
