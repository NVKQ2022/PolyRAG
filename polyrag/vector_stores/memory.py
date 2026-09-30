"""In-memory vector store implementation of BaseVectorStore."""

import math
from typing import Any
import uuid

from polyrag.core.interfaces import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """Zero-dependency in-memory vector store using cosine similarity."""

    def __init__(self) -> None:
        self.vectors: list[list[float]] = []
        self.documents: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.vectors.clear()
        self.documents.clear()

    def add_documents(
        self,
        vectors: list[list[float]],
        documents: list[dict[str, Any]],
        batch_size: int = 5000,
    ) -> None:
        if len(vectors) != len(documents):
            raise ValueError("vectors and documents must have the same length")

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

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not self.vectors:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for vec, doc in zip(self.vectors, self.documents):
            sim = self._cosine_similarity(query_vector, vec)
            scored.append((sim, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, doc in scored[:top_k]:
            results.append(
                {
                    "score": float(score),
                    "distance": float(1.0 - score),
                    "document": doc,
                }
            )
        return results

    def count(self) -> int:
        return len(self.documents)

    def peek(self, limit: int = 5) -> Any:
        return {"documents": [d.get("text", "") for d in self.documents[:limit]]}
