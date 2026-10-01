"""In-memory vector store implementation of BaseVectorStore."""

from collections.abc import Callable
import math
from typing import Any
import uuid

from polyrag.core.interfaces import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """
    Zero-dependency in-memory vector store.

    Supports configurable distance metrics (cosine, l2, dot product)
    and optional filtering by metadata dictionary or custom predicate function.
    """

    def __init__(
        self,
        metric: str = "cosine",
        initial_documents: list[dict[str, Any]] | None = None,
        initial_vectors: list[list[float]] | None = None,
    ) -> None:
        """
        Initialize the in-memory vector store.

        Args:
            metric: Distance metric to use ("cosine", "l2", "dot", "ip"). Defaults to "cosine".
            initial_documents: Optional initial list of documents to seed the store.
            initial_vectors: Optional initial list of vectors matching initial_documents.
        """
        self.metric = metric.lower()
        self.vectors: list[list[float]] = []
        self.documents: list[dict[str, Any]] = []

        if initial_documents or initial_vectors:
            if not initial_documents or not initial_vectors:
                raise ValueError("Both initial_documents and initial_vectors must be provided together.")
            self.add_documents(vectors=initial_vectors, documents=initial_documents)

    def clear(self) -> None:
        """Clear all stored vectors and documents from memory."""
        self.vectors.clear()
        self.documents.clear()

    def add_documents(
        self,
        vectors: list[list[float]],
        documents: list[dict[str, Any]],
        batch_size: int = 5000,
        **kwargs: Any,
    ) -> None:
        """
        Add pre-computed vectors and document records.

        Args:
            vectors: Embedding vectors for each document.
            documents: Document metadata dictionaries containing at least 'text'.
            batch_size: Ignored for in-memory, maintained for interface compliance.
        """
        if len(vectors) != len(documents):
            raise ValueError("vectors and documents must have the same length")
        if not documents:
            return

        for vec, doc in zip(vectors, documents):
            d = dict(doc)
            if "_id" not in d:
                d["_id"] = uuid.uuid4().hex[:8]
            self.vectors.append(vec)
            self.documents.append(d)

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

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
        min_score: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Perform nearest-neighbor search for a query embedding vector.

        Args:
            query_vector: Embedding vector of the query.
            top_k: Maximum number of results to return.
            where: Optional dict of metadata key-value pairs that must match.
            filter_fn: Optional callable predicate returning True for accepted docs.
            min_score: Optional minimum score threshold.
            **kwargs: Extra parameters ignored for compatibility.

        Returns:
            List of scored document matches.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
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

    def count(self) -> int:
        """Return total document count in the vector collection."""
        return len(self.documents)

    def peek(self, limit: int = 5) -> Any:
        """Preview sample records from the vector store."""
        if limit <= 0:
            return {"documents": []}
        return {
            "documents": [d.get("text", "") for d in self.documents[:limit]],
            "ids": [d.get("_id") for d in self.documents[:limit]],
            "records": self.documents[:limit],
        }
