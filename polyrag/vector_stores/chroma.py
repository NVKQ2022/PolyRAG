"""ChromaDB implementation of BaseVectorStore."""

import os
from pathlib import Path
from typing import Any
import uuid

from polyrag.core.interfaces import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of BaseVectorStore.

    Supports:
    - Embedded persistent local storage via chromadb.PersistentClient
    - Remote Chroma server via chromadb.HttpClient
    - Custom distance metrics ("cosine", "l2", "ip") and HNSW parameters
    - Rich metadata and document content filtering (where, where_document)
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        distance_metric: str = "cosine",
        host: str | None = None,
        port: int | None = None,
        ssl: bool = False,
        headers: dict[str, str] | None = None,
        collection_metadata: dict[str, Any] | None = None,
        client: Any | None = None,
        settings: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        """
        Initialize the Chroma vector store. All parameters are optional with smart defaults.

        Args:
            persist_directory: Directory path for local persistent storage.
                Defaults to CHROMA_PERSIST_DIR env var or 'chroma_db'.
            collection_name: Collection name. Defaults to CHROMA_COLLECTION or 'rfc_docs'.
            distance_metric: Distance metric for HNSW index ("cosine", "l2", "ip"). Defaults to "cosine".
            host: Remote Chroma server hostname or IP. If provided, connects via HttpClient.
            port: Remote Chroma server port (default: 8000 when host is specified).
            ssl: Whether to use HTTPS for remote Chroma server connection.
            headers: Optional HTTP headers for remote Chroma server (e.g. auth tokens).
            collection_metadata: Optional dictionary with custom collection configuration or HNSW params.
            client: Optional pre-configured chromadb Client instance.
            settings: Optional chromadb.config.Settings object.
            **client_kwargs: Extra keyword arguments forwarded to the chromadb Client constructor.
        """
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "rfc_docs")
        self.distance_metric = distance_metric.lower()
        self.host = host
        self.port = port
        self.ssl = ssl
        self.headers = headers
        self.collection_metadata = collection_metadata or {}
        self._client = client
        self._settings = settings
        self._client_kwargs = client_kwargs

        self._initialize()

    def _initialize(self) -> None:
        if self._client is not None:
            self.client = self._client
        else:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as err:
                raise ImportError(
                    "chromadb is required to use ChromaVectorStore. "
                    "Install it with `pip install chromadb` or `pip install 'polyrag[chroma]'`."
                ) from err

            active_settings = self._settings or Settings(anonymized_telemetry=False)

            if self.host is not None:
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port or 8000,
                    ssl=self.ssl,
                    headers=self.headers,
                    settings=active_settings,
                    **self._client_kwargs,
                )
            else:
                Path(self.persist_directory).mkdir(
                    parents=True,
                    exist_ok=True,
                )
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=active_settings,
                    **self._client_kwargs,
                )

        merged_metadata = {"hnsw:space": self.distance_metric, **self.collection_metadata}
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=merged_metadata,
        )

    def clear(self) -> None:
        """Drop and recreate the collection, wiping all stored vectors and metadata."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        merged_metadata = {"hnsw:space": self.distance_metric, **self.collection_metadata}
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=merged_metadata,
        )

    def add_documents(
        self,
        vectors: list[list[float]],
        documents: list[dict[str, Any]],
        batch_size: int = 5000,
        **kwargs: Any,
    ) -> None:
        """
        Add pre-computed vectors and document records to ChromaDB.

        Args:
            vectors: List of embedding vectors.
            documents: List of document dictionaries with 'text' and optional metadata.
            batch_size: Maximum items to send per Chroma add batch.
            **kwargs: Extra parameters ignored for compatibility.
        """
        if len(vectors) != len(documents):
            raise ValueError("vectors and documents must have the same length")
        if not documents:
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        texts = [doc["text"] for doc in documents]
        metadatas = [{k: v for k, v in doc.items() if k != "text"} for doc in documents]

        ids = [
            str(
                doc.get("_id")
                or (
                    f"{Path(str(doc.get('source', 'doc'))).stem}_"
                    f"{doc.get('chunk_id', 0)}_"
                    f"{uuid.uuid4().hex[:8]}"
                )
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
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Perform nearest-neighbor search for a query embedding vector.

        Args:
            query_vector: Embedding vector of the query.
            top_k: Number of nearest neighbors to retrieve.
            where: Optional Chroma metadata filter expression, e.g. {"source": "rfc1035.txt"}.
            where_document: Optional document content filter, e.g. {"$contains": "DNS"}.
            include: Optional list of fields to include (defaults to documents, metadatas, distances).
            **kwargs: Extra search arguments passed to Chroma collection.query().

        Returns:
            List of scored document matches.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_args: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": include or ["documents", "metadatas", "distances"],
            **kwargs,
        }
        if where:
            query_args["where"] = where
        if where_document:
            query_args["where_document"] = where_document

        results = self.collection.query(**query_args)

        if not results.get("ids") or not results["ids"][0]:
            return []

        output: list[dict[str, Any]] = []
        for document_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            dist = float(distance)
            if self.distance_metric == "l2":
                score = 1.0 / (1.0 + dist)
            elif self.distance_metric == "ip":
                score = dist
            else:  # default cosine
                score = 1.0 - dist

            output.append(
                {
                    "score": float(score),
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
        """Return total document count in the vector collection."""
        return self.collection.count()

    def peek(self, limit: int = 5) -> Any:
        """Preview sample records from the vector store."""
        if limit <= 0:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.collection.peek(limit)
