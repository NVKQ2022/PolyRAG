"""ChromaDB implementation of BaseVectorStore."""

from pathlib import Path
from typing import Any
import uuid

from polyrag.core.interfaces import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of BaseVectorStore."""

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "rfc_docs",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._initialize()

    def _initialize(self) -> None:
        import chromadb
        from chromadb.config import Settings

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        vectors: list[list[float]],
        documents: list[dict[str, Any]],
        batch_size: int = 5000,
    ) -> None:
        if not documents:
            return
        if len(vectors) != len(documents):
            raise ValueError("vectors and documents must have the same length")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        texts = [doc["text"] for doc in documents]
        metadatas = [{k: v for k, v in doc.items() if k != "text"} for doc in documents]

        ids = [
            (
                f"{Path(str(doc.get('source', 'doc'))).stem}_"
                f"{doc.get('chunk_id', 0)}_"
                f"{uuid.uuid4().hex[:8]}"
            )
            for doc in documents
        ]

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectors[start:end],
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        output: list[dict[str, Any]] = []
        for document_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            dist = float(distance)
            output.append(
                {
                    "score": 1.0 - dist,
                    "distance": dist,
                    "document": {
                        "_id": document_id,
                        "text": text,
                        **(metadata or {}),
                    },
                }
            )

        return output

    def count(self) -> int:
        return self.collection.count()

    def peek(self, limit: int = 5) -> Any:
        return self.collection.peek(limit)
