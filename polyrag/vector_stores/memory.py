"""In-memory Vector Store conforming to LangChain VectorStore and PolyRAG BaseVectorStore."""

import math
from typing import Any, Callable, Sequence
import uuid

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from polyrag.core.interfaces import BaseVectorStore
from polyrag.core.models import Document


class InMemoryVectorStore(BaseVectorStore):
    """
    Zero-dependency in-memory vector store conforming to LangChain VectorStore.
    Supports cosine, Euclidean (L2), and dot-product similarity metrics,
    dictionary-based metadata filtering, and custom predicate filter functions.
    """

    def __init__(
        self,
        embedding: Embeddings | Any | None = None,
        metric: str = "cosine",
        initial_documents: list[dict[str, Any]] | None = None,
        initial_vectors: list[list[float]] | None = None,
        **kwargs: Any,
    ) -> None:
        if metric not in ("cosine", "l2", "dot", "ip"):
            raise ValueError(f"Unsupported metric: {metric!r}. Use 'cosine', 'l2', or 'dot'.")
        self.metric = metric
        self.embedding = embedding or FakeEmbeddings(size=384)
        self.embedding_model = self.embedding

        self.vectors: list[list[float]] = []
        self.documents: list[dict[str, Any]] = []

        if initial_documents is not None or initial_vectors is not None:
            if not initial_documents or not initial_vectors:
                raise ValueError("Both initial_documents and initial_vectors must be provided together.")
            self.add_documents(vectors=initial_vectors, documents=initial_documents)

    def clear(self) -> None:
        """Clear all stored vectors and documents from memory."""
        self.vectors.clear()
        self.documents.clear()

    def count(self) -> int:
        """Return total document count in the vector collection."""
        return len(self.documents)

    def peek(self, limit: int = 5) -> dict[str, Any]:
        """Preview sample records from the vector store."""
        if limit <= 0:
            return {"documents": [], "ids": [], "records": []}
        return {
            "documents": [d.get("text", d.get("page_content", "")) for d in self.documents[:limit]],
            "ids": [d.get("_id", d.get("id")) for d in self.documents[:limit]],
            "records": self.documents[:limit],
        }

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _euclidean_distance(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _dot_product(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def add_documents(
        self,
        documents: list[Any] | None = None,
        vectors: list[list[float]] | None = None,
        batch_size: int = 5000,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Add documents and optional vectors to the store.

        Supports:
        - PolyRAG legacy positional: add_documents(vectors, documents)
        - PolyRAG keyword: add_documents(vectors=vectors, documents=documents)
        - LangChain standard: add_documents(documents=[Document(...)])
        """
        # Handle positional swap: add_documents(vectors, documents)
        if documents is not None and isinstance(documents, list) and len(documents) > 0:
            if isinstance(documents[0], list) and (
                vectors is not None and isinstance(vectors, list) and len(vectors) > 0 and isinstance(vectors[0], dict)
            ):
                vectors, documents = documents, vectors  # type: ignore

        if not documents:
            return []

        # If vectors were not passed, generate them via self.embedding
        if vectors is None:
            texts: list[str] = []
            for d in documents:
                if isinstance(d, LCDocument):
                    texts.append(d.page_content)
                elif isinstance(d, dict):
                    texts.append(str(d.get("text", d.get("page_content", ""))))
                else:
                    texts.append(str(d))

            if hasattr(self.embedding, "embed_documents"):
                vectors = self.embedding.embed_documents(texts)
            elif hasattr(self.embedding, "embed_batch"):
                vectors = self.embedding.embed_batch(texts)
            else:
                vectors = [[0.0] * 384 for _ in texts]

        if len(vectors) != len(documents):
            raise ValueError(f"vectors ({len(vectors)}) and documents ({len(documents)}) must have the same length")

        generated_ids: list[str] = []
        for idx, (vec, doc) in enumerate(zip(vectors, documents)):
            if isinstance(doc, LCDocument):
                d = dict(doc.metadata)
                d["text"] = doc.page_content
                d["page_content"] = doc.page_content
                doc_id = doc.id or d.get("_id") or (ids[idx] if ids and idx < len(ids) else uuid.uuid4().hex[:8])
                d["_id"] = doc_id
            elif isinstance(doc, dict):
                d = dict(doc)
                doc_id = d.get("_id") or d.get("id") or (ids[idx] if ids and idx < len(ids) else uuid.uuid4().hex[:8])
                d["_id"] = doc_id
                if "text" not in d and "page_content" in d:
                    d["text"] = d["page_content"]
                elif "text" in d and "page_content" not in d:
                    d["page_content"] = d["text"]
            else:
                doc_id = ids[idx] if ids and idx < len(ids) else uuid.uuid4().hex[:8]
                d = {"_id": doc_id, "text": str(doc), "page_content": str(doc)}

            self.vectors.append(vec)
            self.documents.append(d)
            generated_ids.append(str(doc_id))

        return generated_ids

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """LangChain standard: add list of raw text strings."""
        docs = [
            Document(page_content=t, metadata=(metadatas[i] if metadatas and i < len(metadatas) else {}))
            for i, t in enumerate(texts)
        ]
        return self.add_documents(documents=docs, **kwargs)

    def search(
        self,
        query_vector: list[float] | None = None,
        query: str = "",
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
        min_score: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Perform nearest-neighbor search for a query embedding vector or text.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if query_vector is None:
            if not query:
                return []
            if hasattr(self.embedding, "embed_query"):
                query_vector = self.embedding.embed_query(query)
            elif hasattr(self.embedding, "embed_text"):
                query_vector = self.embedding.embed_text(query)
            else:
                return []

        if not self.vectors:
            return []

        candidates: list[tuple[float, float, dict[str, Any]]] = []

        for vec, doc in zip(self.vectors, self.documents):
            # Apply where filter
            if where:
                match = True
                for k, v in where.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Apply filter_fn
            if filter_fn is not None and not filter_fn(doc):
                continue

            if self.metric == "l2":
                dist = self._euclidean_distance(query_vector, vec)
                score = 1.0 / (1.0 + dist)
            elif self.metric in ("dot", "ip"):
                score = self._dot_product(query_vector, vec)
                dist = 1.0 - score
            else:  # default "cosine"
                sim = self._cosine_similarity(query_vector, vec)
                score = sim
                dist = 1.0 - sim

            if min_score is not None and score < min_score:
                continue

            candidates.append((score, dist, doc))

        candidates.sort(key=lambda item: item[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, dist, doc in candidates[:top_k]:
            results.append(
                {
                    "score": float(score),
                    "distance": float(dist),
                    "document": doc,
                }
            )
        return results

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[LCDocument]:
        """LangChain standard: search by embedding vector returning Document objects."""
        hits = self.search(query_vector=embedding, top_k=k, **kwargs)
        return [
            Document(
                page_content=r["document"].get("text", r["document"].get("page_content", "")),
                metadata=r["document"],
                id=r["document"].get("_id"),
            )
            for r in hits
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[LCDocument, float]]:
        """LangChain standard: search by vector returning (Document, score) tuples."""
        hits = self.search(query_vector=embedding, top_k=k, **kwargs)
        return [
            (
                Document(
                    page_content=r["document"].get("text", r["document"].get("page_content", "")),
                    metadata=r["document"],
                    id=r["document"].get("_id"),
                ),
                r["score"],
            )
            for r in hits
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[LCDocument]:
        """LangChain standard: search by query string returning Document objects."""
        hits = self.search(query=query, top_k=k, **kwargs)
        return [
            Document(
                page_content=r["document"].get("text", r["document"].get("page_content", "")),
                metadata=r["document"],
                id=r["document"].get("_id"),
            )
            for r in hits
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[LCDocument, float]]:
        """LangChain standard: search by query string returning (Document, score) tuples."""
        hits = self.search(query=query, top_k=k, **kwargs)
        return [
            (
                Document(
                    page_content=r["document"].get("text", r["document"].get("page_content", "")),
                    metadata=r["document"],
                    id=r["document"].get("_id"),
                ),
                r["score"],
            )
            for r in hits
        ]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "InMemoryVectorStore":
        """LangChain standard: instantiate and populate from raw texts."""
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store


__all__ = ["InMemoryVectorStore"]
