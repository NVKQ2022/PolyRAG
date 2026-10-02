# Milvus & Milvus Lite Vector Stores (`polyrag.vector_stores.milvus`)

The `polyrag.vector_stores` module provides enterprise-grade, highly scalable vector indexing powered by **Milvus** via the modern `pymilvus.MilvusClient` SDK.

PolyRAG includes two specialized vector store classes:
1. **`MilvusLiteVectorStore` (alias `MilvusLite`)**: Embedded, serverless local database file (zero external dependencies, zero Docker, runs directly in Python).
2. **`MilvusVectorStore`**: Full client for standalone Docker instances, distributed Kubernetes clusters, and managed **Zilliz Cloud**.

Both classes implement PolyRAG's unified [`BaseVectorStore`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/core/interfaces.py) contract, allowing seamless transition from a local prototype to a planetary-scale distributed cluster with zero code modifications.

---

## 1. Installation

Install PolyRAG with the `[milvus]` extra:

```bash
pip install "polyrag[milvus]"
```

Or install `pymilvus` directly:

```bash
pip install "pymilvus>=2.4.0"
```

---

## 2. Quickstart: Milvus Lite vs. Milvus Cluster

### Option A: Embedded Milvus Lite (Recommended for Local Dev & Edge)
Milvus Lite runs inside your Python process and stores data in a single local database file. No Docker or external service required:

```python
from polyrag.vector_stores import MilvusLiteVectorStore

# Automatically creates parent directories if needed
vdb = MilvusLiteVectorStore(
    db_path="./data/milvus_demo.db",
    collection_name="kb_articles",
    metric_type="COSINE",
)
```

Alternatively, use the convenience alias or the factory method:
```python
from polyrag.vector_stores import MilvusLite, MilvusVectorStore

# Using alias
vdb = MilvusLite(db_path="./data/milvus_demo.db")

# Or via factory method
vdb = MilvusVectorStore.lite(db_path="./data/milvus_demo.db")

# In-memory ephemeral database for fast testing
mem_vdb = MilvusLiteVectorStore(db_path=":memory:")
```

### Option B: Milvus Standalone (Docker)
Run Milvus locally with Docker:
```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.4.0 milvus run standalone
```

Connect via PolyRAG:
```python
from polyrag.vector_stores import MilvusVectorStore

vdb = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="production_kb",
)
```

### Option C: Zilliz Cloud (Fully Managed)
```python
from polyrag.vector_stores import MilvusVectorStore

vdb = MilvusVectorStore(
    uri="https://in03-xxxxxxxx.api.gcp-us-west1.zillizcloud.com",
    token="YOUR_ZILLIZ_API_KEY",
    collection_name="production_kb",
)
```

---

## 3. `MilvusLiteVectorStore` Reference

`MilvusLiteVectorStore` inherits directly from `MilvusVectorStore` and specializes it for file-based embedded storage:

```python
MilvusLiteVectorStore(
    db_path: str | Path = "./milvus_lite.db",
    collection_name: str | None = None,
    dimension: int | None = None,
    metric_type: str = "COSINE",
    index_type: str = "AUTOINDEX",
    index_params: dict[str, Any] | None = None,
    search_params: dict[str, Any] | None = None,
    partition_name: str | None = None,
    consistency_level: str = "Strong",
    timeout: float | None = None,
    output_fields: list[str] | None = None,
    client: Any | None = None,
    **client_kwargs: Any,
)
```

### Key Lite Features
* **`db_path`**: Accepts a file path (`str` or `pathlib.Path`, e.g. `'./milvus_lite.db'`, `'data/vectors.db'`) or `':memory:'` for in-memory ephemeral testing.
* **Auto-Directory Creation**: Nested directories specified in `db_path` (e.g. `'storage/nested/db.milvus'`) are automatically created recursively.
* **Zero Infrastructure**: Milvus Lite is built into `pymilvus>=2.4.0` via an embedded C++ core. No external daemon or network port is used.
* **Seamless Scalability**: Once your dataset grows beyond single-machine capacity (~1M vectors), swap `MilvusLiteVectorStore(db_path=...)` with `MilvusVectorStore(uri="http://...")` without changing any indexing, query, or pipeline logic.

---

## 4. `MilvusVectorStore` Reference

```python
MilvusVectorStore(
    uri: str | None = None,
    token: str | None = None,
    collection_name: str | None = None,
    db_name: str = "default",
    dimension: int | None = None,
    metric_type: str = "COSINE",
    index_type: str = "AUTOINDEX",
    index_params: dict[str, Any] | None = None,
    search_params: dict[str, Any] | None = None,
    partition_name: str | None = None,
    consistency_level: str = "Strong",
    timeout: float | None = None,
    output_fields: list[str] | None = None,
    client: Any | None = None,
    **client_kwargs: Any,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uri` | `str \| None` | `os.getenv("MILVUS_URI", "http://localhost:19530")` | Endpoint URL or local database file path (`./milvus.db`) for Milvus Lite. |
| `token` | `str \| None` | `os.getenv("MILVUS_TOKEN", "")` | API key or token for Zilliz Cloud or authentication (`user:password`). |
| `collection_name` | `str \| None` | `os.getenv("MILVUS_COLLECTION", "polyrag_docs")` | Target Milvus collection. |
| `db_name` | `str` | `"default"` | Database name for multi-tenant deployments (Milvus 2.3+). |
| `dimension` | `int \| None` | `None` | Embedding vector dimension. Auto-inferred on first `add_documents` call if omitted. |
| `metric_type` | `str` | `"COSINE"` | Similarity metric: `"COSINE"`, `"L2"`, or `"IP"`. |
| `index_type` | `str` | `"AUTOINDEX"` | Vector index type: `"AUTOINDEX"`, `"HNSW"`, `"IVF_FLAT"`, or `"FLAT"`. |
| `index_params` | `dict \| None` | `None` | Custom index parameters (e.g. `{"M": 16, "efConstruction": 200}`). |
| `search_params` | `dict \| None` | `None` | Default search parameters applied to all queries (e.g. `{"params": {"nprobe": 10}}`). |
| `partition_name` | `str \| None` | `None` | Default partition to insert into and query from for multi-tenant isolation. |
| `consistency_level`| `str` | `"Strong"` | Consistency model: `"Strong"`, `"Bounded"`, `"Session"`, or `"Eventually"`. |
| `timeout` | `float \| None`| `None` | Default RPC timeout in seconds for collection, search, and insertion operations. |
| `output_fields` | `list[str] \| None`| `None` | Default scalar fields to retrieve on search queries. |
| `client` | `Any \| None` | `None` | Pre-configured `MilvusClient` instance (useful for dependency injection or mocking). |
| `**client_kwargs` | `Any` | `{}` | Additional kwargs forwarded directly to `MilvusClient(...)`. |

---

## 5. Key Architectural Features

### Dynamic Schema & Non-Destructive Ingestion
Both `MilvusVectorStore` and `MilvusLiteVectorStore` create collections with `enable_dynamic_field=True`. This allows arbitrary metadata fields (`source`, `chunk_id`, `author`, `department`, custom tags) to be stored and searched without requiring database schema migrations.

### Distance Metrics & Normalization
* **`COSINE`**: Milvus returns the cosine similarity score in `[-1, 1]`.
  * `score = distance`
  * `distance = 1.0 - score`
* **`L2`**: Milvus returns the Euclidean distance `>= 0`.
  * `score = 1.0 / (1.0 + distance)`
* **`IP` (Inner Product)**: Returns the raw dot product (ideal for normalized vectors).

### Robust Two-Stage Counting
`store.count()` executes an accurate `query(output_fields=["count(*)"])` reflecting real-time inserts and deletions. If the collection is not loaded in memory, it falls back to segment-level metadata stats (`get_collection_stats()`).

---

## 6. Method Reference

### `add_documents(vectors, documents, batch_size=5000)`
Inserts embeddings and document payloads into the Milvus collection.

```python
vectors = [
    [0.05, 0.92, 0.31, ...],  # Dim: 384
    [0.88, 0.12, 0.04, ...],
]

documents = [
    {
        "_id": "kb_auth_001",
        "text": "Article ID: KB-AUTH-001\nSymptoms: Invalid token message...",
        "source": "kb/auth.md",
        "category": "Authentication",
        "module": "Customer Portal",
    },
    {
        "_id": "kb_net_002",
        "text": "Article ID: KB-NET-002\nSymptoms: DNS lookup timeout...",
        "source": "kb/network.md",
        "category": "Network",
    },
]

vdb.add_documents(vectors=vectors, documents=documents, batch_size=2000)
```

---

### `search(query_vector, top_k=5) -> list[dict[str, Any]]`
Searches top-$k$ nearest neighbors:

```python
query_vec = [0.06, 0.90, 0.30, ...]
results = vdb.search(query_vector=query_vec, top_k=2)

for hit in results:
    print(f"Score: {hit['score']:.4f} | Distance: {hit['distance']:.4f}")
    print(f"ID:    {hit['document']['_id']}")
    print(f"Text:  {hit['document']['text'][:80]}...")
    print(f"Category: {hit['document'].get('category')}")
```

#### Output Structure
```python
[
    {
        "score": 0.9842,
        "distance": 0.0158,
        "document": {
            "_id": "kb_auth_001",
            "text": "Article ID: KB-AUTH-001\nSymptoms: Invalid token message...",
            "source": "kb/auth.md",
            "category": "Authentication",
            "module": "Customer Portal",
        },
    }
]
```

---

### `count() -> int`
Returns total number of entities in the collection:

```python
total = vdb.count()
print(f"Total entities: {total}")
```

---

### `peek(limit=5) -> Any`
Previews sample entities:

```python
preview = vdb.peek(limit=3)
print(preview["ids"])
print(preview["documents"])
print(preview["metadatas"])
```

---

### `clear() -> None`
Drops and recreates the collection:

```python
vdb.clear()
assert vdb.count() == 0
```

---

## 7. End-to-End Example with PolyRAG

```python
from polyrag.app import PolyRAG
from polyrag.vector_stores import MilvusLiteVectorStore
from polyrag.embeddings import SentenceTransformerEmbedding
from polyrag.llms import OpenAILLM

# 1. Initialize embedded Milvus Lite
vdb = MilvusLiteVectorStore(
    db_path="./storage/rag_knowledge.db",
    collection_name="enterprise_knowledge",
    metric_type="COSINE",
)

# 2. Wire into PolyRAG
rag = PolyRAG.create(
    vector_store=vdb,
    embedding_model=SentenceTransformerEmbedding("all-MiniLM-L6-v2"),
    llm_client=OpenAILLM(model_name="gpt-4o-mini"),
)

# 3. Ingest documents
rag.ingest_directory("docs/", glob_pattern="**/*.md")

# 4. Use Advanced RAG (Multi-Query Expansion + RRF Re-Ranking)
advanced = rag.create_advanced_rag(top_k=5, num_expanded_queries=3)
response = advanced.query("How do I handle token expiration in Customer Portal?")

print("Answer:", response.answer)
print("Confidence:", response.confidence)
```

---

## 8. Production Best Practices

* **Batch Sizing**: When indexing tens of thousands of chunks, use `batch_size=2000` to `batch_size=5000` for optimal network throughput and memory utilization.
* **Consistency Level**:
  * Use `"Strong"` (default) for immediate read-your-writes guarantees (ideal during ingestion and testing).
  * Use `"Bounded"` or `"Session"` for high-throughput distributed read clusters.
* **Milvus Lite vs Standalone**: Use `MilvusLiteVectorStore` for single-server setups, edge deployments, and testing. Migrate to `MilvusVectorStore` (Standalone or Cluster) when your vector count exceeds ~1M vectors or requires multi-node high availability.
