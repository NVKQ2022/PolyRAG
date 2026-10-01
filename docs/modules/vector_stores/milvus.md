# MilvusVectorStore (`polyrag.vector_stores.milvus`)

The `MilvusVectorStore` provides enterprise-grade, highly scalable vector indexing powered by **Milvus** via the modern `pymilvus.MilvusClient` SDK.

It supports **Milvus Lite** (local embedded database file), **Milvus Standalone** (single Docker container), **Distributed Milvus Clusters** (Kubernetes), and **Zilliz Cloud** (fully managed cloud vector database).

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

## 2. Deployment Targets

### Option A: Embedded Milvus Lite (Zero Infrastructure)
Milvus Lite compiles directly into your Python process and stores data in a single local database file. No Docker or external service required:

```python
from polyrag.vector_stores import MilvusVectorStore

vdb = MilvusVectorStore(
    uri="./milvus_demo.db",
    collection_name="kb_articles",
    metric_type="COSINE",
)
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
vdb = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="production_kb",
)
```

### Option C: Zilliz Cloud (Managed)
```python
vdb = MilvusVectorStore(
    uri="https://in03-xxxxxxxx.api.gcp-us-west1.zillizcloud.com",
    token="YOUR_ZILLIZ_API_KEY",
    collection_name="production_kb",
)
```

---

## 3. Constructor & Parameters

```python
MilvusVectorStore(
    uri: str = "http://localhost:19530",
    token: str = "",
    collection_name: str = "polyrag_docs",
    dimension: int | None = None,
    metric_type: str = "COSINE",
    consistency_level: str = "Strong",
    client: Any | None = None,
    **client_kwargs: Any,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uri` | `str` | `"http://localhost:19530"` | Endpoint URL (for Standalone/Cluster/Cloud) or file path (`./milvus.db`) for Milvus Lite. |
| `token` | `str` | `""` | API key or token for Zilliz Cloud or basic authentication (`user:password`). |
| `collection_name` | `str` | `"polyrag_docs"` | Target Milvus collection. |
| `dimension` | `int \| None` | `None` | Embedding dimension. If omitted (`None`), it is **automatically inferred** on the first call to `add_documents`. |
| `metric_type` | `str` | `"COSINE"` | Similarity metric: `"COSINE"`, `"L2"`, or `"IP"`. |
| `consistency_level` | `str` | `"Strong"` | Consistency model: `"Strong"`, `"Bounded"`, `"Session"`, or `"Eventually"`. |
| `client` | `MilvusClient \| None` | `None` | Pre-configured `MilvusClient` instance (useful for dependency injection or mocking in tests). |
| `**client_kwargs` | `Any` | `{}` | Additional kwargs forwarded to `MilvusClient(...)`. |

---

## 4. Key Architectural Features

### Dynamic Schema & Non-Destructive Ingestion
`MilvusVectorStore` creates collections with `enable_dynamic_field=True`. This allows arbitrary metadata fields (`source`, `chunk_id`, `author`, `department`, custom tags) to be stored and searched without requiring database schema migrations.

### Distance Metrics & Normalization
* **`COSINE`**: Milvus returns the cosine similarity score $\in [-1, 1]$.
  * $\text{score} = \text{distance}$
  * $\text{distance} = 1.0 - \text{score}$
* **`L2`**: Milvus returns the Euclidean distance $\ge 0$.
  * $\text{score} = \frac{1}{1.0 + \text{distance}}$
* **`IP` (Inner Product)**: Returns the raw dot product (ideal for normalized vectors).

### Robust Two-Stage Counting
`MilvusVectorStore.count()` executes an accurate `query(output_fields=["count(*)"])` reflecting real-time inserts and deletions. If the collection is not loaded in memory, it falls back to segment-level metadata stats (`get_collection_stats()`).

---

## 5. Method Reference

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

## 6. End-to-End Example with PolyRAG

```python
from polyrag.app import PolyRAG
from polyrag.vector_stores import MilvusVectorStore
from polyrag.embeddings import SentenceTransformerEmbedding
from polyrag.llms import OpenAILLM

# 1. Initialize Milvus (Lite or Server)
vdb = MilvusVectorStore(
    uri="./production_kb.db",
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

## 7. Production Best Practices

* **Batch Sizing**: When indexing tens of thousands of chunks, use `batch_size=2000` to `batch_size=5000` for optimal network throughput and memory utilization.
* **Consistency Level**:
  * Use `"Strong"` (default) for immediate read-your-writes guarantees (ideal during ingestion and testing).
  * Use `"Bounded"` or `"Session"` for high-throughput distributed read clusters.
* **Milvus Lite vs Standalone**: Use Milvus Lite for single-server setups, edge deployments, and testing. Migrate to Milvus Standalone / Cluster when your vector count exceeds $\sim 1\text{M}$ vectors or requires multi-node high availability.
