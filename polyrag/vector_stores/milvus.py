"""Milvus vector store implementation of BaseVectorStore."""

from pathlib import Path
from typing import Any
import uuid

from polyrag.core.interfaces import BaseVectorStore


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus implementation of BaseVectorStore using pymilvus.MilvusClient.

    Supports:
    - Milvus Lite (local file uri, e.g., 'milvus_demo.db' or './milvus.db')
    - Milvus Standalone / Distributed cluster (e.g., 'http://localhost:19530')
    - Zilliz Cloud (e.g., 'https://in03-...zillizcloud.com', token='...')
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        collection_name: str = "polyrag_docs",
        dimension: int | None = None,
        metric_type: str = "COSINE",
        consistency_level: str = "Strong",
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        """
        Initialize the Milvus vector store.

        Args:
            uri: Milvus endpoint URI or local SQLite-style file path for Milvus Lite.
            token: Authentication token (e.g., for Zilliz Cloud or user:password).
            collection_name: Target Milvus collection name.
            dimension: Dimensionality of vector embeddings. If None, will be inferred
                automatically on the first call to add_documents.
            metric_type: Distance metric type ("COSINE", "L2", "IP"). Defaults to "COSINE".
            consistency_level: Consistency level ("Strong", "Bounded", "Session", "Eventually").
            client: Optional pre-configured MilvusClient instance.
            **client_kwargs: Extra keyword arguments forwarded to MilvusClient.
        """
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type.upper()
        self.consistency_level = consistency_level
        self._client = client
        self._client_kwargs = client_kwargs

        self._ensure_client()
        if self.dimension is not None:
            self._ensure_collection(self.dimension)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from pymilvus import MilvusClient
        except ImportError as err:
            raise ImportError(
                "pymilvus is required to use MilvusVectorStore. "
                "Install it with `pip install pymilvus` or `pip install 'polyrag[milvus]'`."
            ) from err

        kwargs = dict(self._client_kwargs)
        if self.token:
            kwargs["token"] = self.token

        self._client = MilvusClient(uri=self.uri, **kwargs)
        return self._client

    @property
    def client(self) -> Any:
        """Return the underlying MilvusClient instance."""
        return self._ensure_client()

    def _collection_exists(self) -> bool:
        try:
            return bool(self.client.has_collection(collection_name=self.collection_name))
        except Exception:
            return False

    def _ensure_collection(self, dimension: int) -> None:
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                primary_field_name="id",
                id_type="string",
                max_length=512,
                auto_id=False,
                metric_type=self.metric_type,
                enable_dynamic_field=True,
                consistency_level=self.consistency_level,
            )

    def clear(self) -> None:
        """Clear all records by dropping and recreating the collection."""
        if self._collection_exists():
            self.client.drop_collection(collection_name=self.collection_name)
        if self.dimension is not None:
            self._ensure_collection(self.dimension)

    def add_documents(
        self,
        vectors: list[list[float]],
        documents: list[dict[str, Any]],
        batch_size: int = 5000,
    ) -> None:
        """
        Add pre-computed vectors and document records to the collection.

        Args:
            vectors: List of embedding vectors.
            documents: List of document records containing 'text' and optional metadata.
            batch_size: Number of entities to insert per batch.
        """
        if len(vectors) != len(documents):
            raise ValueError("vectors and documents must have the same length")
        if not documents:
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        dim = len(vectors[0])
        if self.dimension is None:
            self.dimension = dim
        self._ensure_collection(dim)

        rows: list[dict[str, Any]] = []
        for vec, doc in zip(vectors, documents):
            doc_id = str(
                doc.get("_id")
                or (
                    f"{Path(str(doc.get('source', 'doc'))).stem}_"
                    f"{doc.get('chunk_id', 0)}_"
                    f"{uuid.uuid4().hex[:8]}"
                )
            )
            text = doc.get("text", "")
            metadata = {k: v for k, v in doc.items() if k not in ("text", "_id")}

            row: dict[str, Any] = {
                "id": doc_id,
                "vector": vec,
                "text": text,
                "metadata": metadata,
            }
            # Populate top-level fields for dynamic filtering in Milvus
            for k, v in metadata.items():
                if k not in row:
                    row[k] = v

            rows.append(row)

        for start in range(0, len(rows), batch_size):
            end = start + batch_size
            self.client.insert(
                collection_name=self.collection_name,
                data=rows[start:end],
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Perform nearest-neighbor search for a query embedding vector.

        Args:
            query_vector: Embedding vector of the query.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of dicts with 'score', 'distance', and 'document'.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not self._collection_exists():
            return []

        try:
            search_res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["*"],
            )
        except Exception:
            search_res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["text", "metadata"],
            )

        if not search_res or not search_res[0]:
            return []

        output: list[dict[str, Any]] = []
        for hit in search_res[0]:
            hit_dict = hit if isinstance(hit, dict) else hit.to_dict()
            doc_id = hit_dict.get("id")
            raw_dist = float(hit_dict.get("distance", 0.0))
            entity = hit_dict.get("entity", {})

            if self.metric_type in ("COSINE", "IP"):
                score = raw_dist
                distance = 1.0 - raw_dist
            else:  # L2 or other distance metrics
                distance = raw_dist
                score = 1.0 / (1.0 + raw_dist)

            text = entity.get("text", "")
            meta = entity.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}

            # Collect any dynamic fields not already in meta
            extra = {
                k: v
                for k, v in entity.items()
                if k not in ("text", "metadata", "id", "vector")
            }
            combined_meta = {**meta, **extra}

            output.append(
                {
                    "score": float(score),
                    "distance": float(distance),
                    "document": {
                        "_id": doc_id,
                        "text": text,
                        **combined_meta,
                    },
                }
            )

        return output

    def count(self) -> int:
        """Return total document count in the vector collection."""
        if not self._collection_exists():
            return 0

        # Try accurate query count(*) first
        try:
            res = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["count(*)"],
            )
            if res and isinstance(res, list) and len(res) > 0:
                first = res[0]
                if isinstance(first, dict) and "count(*)" in first:
                    return int(first["count(*)"])
        except Exception:
            pass

        # Fallback to collection stats
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def peek(self, limit: int = 5) -> Any:
        """Preview sample records from the vector store."""
        if limit <= 0 or not self._collection_exists():
            return {"documents": []}

        try:
            res = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["id", "text", "metadata"],
                limit=limit,
            )
            return {
                "documents": [r.get("text", "") for r in res],
                "ids": [r.get("id") for r in res],
                "metadatas": [r.get("metadata", {}) for r in res],
            }
        except Exception:
            return {"documents": []}
