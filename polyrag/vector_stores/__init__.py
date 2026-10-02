"""Vector Stores Package for polyrag built on langchain_core.vectorstores."""

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

    Args:
        vector_store: VectorStore instance or string name ('memory', 'chroma', 'milvus', 'milvus_lite').
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
            from polyrag.vector_stores.chroma import ChromaVectorStore
            return ChromaVectorStore(**kwargs)
        if normalized in ("milvus_lite", "lite"):
            from polyrag.vector_stores.milvus import MilvusLiteVectorStore
            return MilvusLiteVectorStore(**kwargs)
        if normalized == "milvus":
            from polyrag.vector_stores.milvus import MilvusVectorStore
            return MilvusVectorStore(**kwargs)

    raise TypeError(
        f"Expected VectorStore instance or recognized store name, got {type(vector_store).__name__}"
    )


def __getattr__(name: str) -> Any:
    if name == "ChromaVectorStore":
        from polyrag.vector_stores.chroma import ChromaVectorStore
        return ChromaVectorStore
    if name in ("MilvusVectorStore", "MilvusLiteVectorStore", "MilvusLite"):
        from polyrag.vector_stores import milvus
        return getattr(milvus, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseVectorStore",
    "VectorStore",
    "InMemoryVectorStore",
    "resolve_vector_store",
    "ChromaVectorStore",
    "MilvusVectorStore",
    "MilvusLiteVectorStore",
    "MilvusLite",
]
