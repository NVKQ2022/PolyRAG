# Module: `polyrag.vector_stores` 🗄️

The `polyrag.vector_stores` module manages nearest-neighbor vector indexing, document persistence, and semantic similarity search.

---

## 1. Available Vector Stores

| Vector Store | Dependencies | Persistence | Best Used For |
| :--- | :--- | :--- | :--- |
| **`InMemoryVectorStore`** | Zero dependencies | RAM only (ephemeral) | Unit tests, quick prototypes, air-gapped dev |
| **`ChromaVectorStore`** | `chromadb>=0.4.0` | Disk / Local SQLite | Embedded local apps, desktop tools |
| **`MilvusVectorStore`** | `pymilvus>=2.4.0` | Milvus Lite file, Server, or Cloud | Production scale, enterprise clusters, Zilliz Cloud |

---

## 2. `InMemoryVectorStore`

Included in the core package with **zero dependencies**. Computes exact cosine similarity across all stored vectors.

```python
from polyrag.vector_stores import InMemoryVectorStore

vdb = InMemoryVectorStore()

# Standard BaseVectorStore operations
vdb.add_documents(
    vectors=[[0.1, 0.2], [0.8, 0.9]],
    documents=[{"text": "Chunk 1", "source": "a.txt"}, {"text": "Chunk 2", "source": "b.txt"}],
)

results = vdb.search(query_vector=[0.1, 0.2], top_k=1)
print(f"Top match: {results[0]['document']['text']}")
print(f"Total count: {vdb.count()}")
```

---

## 3. `ChromaVectorStore`

Embeds a local persistent Chroma collection using HNSW indexing and cosine distance.

### Installation
```bash
pip install "polyrag[chroma]"
```

### Usage
```python
from polyrag.vector_stores import ChromaVectorStore

vdb = ChromaVectorStore(
    persist_directory="chroma_storage",
    collection_name="knowledge_base",
)
```

---

## 4. `MilvusVectorStore`

Provides production-grade vector storage powered by `pymilvus.MilvusClient`.

### Installation
```bash
pip install "polyrag[milvus]"
```

### Modes of Operation

#### Mode A: Embedded Milvus Lite (Zero Server Setup)
Milvus Lite embeds directly into your Python process using a local SQLite-like database file:
```python
from polyrag.vector_stores import MilvusVectorStore

vdb = MilvusVectorStore(
    uri="./milvus_demo.db",
    collection_name="kb_docs",
    metric_type="COSINE",  # 'COSINE', 'L2', or 'IP'
)
```

#### Mode B: Milvus Standalone or Distributed Cluster
```python
vdb = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="production_kb",
)
```

#### Mode C: Zilliz Cloud (Fully Managed)
```python
vdb = MilvusVectorStore(
    uri="https://in03-xxxxxxxx.api.gcp-us-west1.zillizcloud.com",
    token="YOUR_ZILLIZ_API_KEY",
    collection_name="production_kb",
)
```

### Key Parameters
* `uri`: Endpoint URI (`http://...`) or file path (`./milvus.db`).
* `token`: API key or authentication token for cloud clusters.
* `collection_name`: Target collection name (default: `"polyrag_docs"`).
* `dimension`: Optional vector dimension. Automatically inferred on first `add_documents` if omitted.
* `metric_type`: Distance metric: `"COSINE"` (default), `"L2"`, or `"IP"`.
* `consistency_level`: Consistency model (`"Strong"`, `"Bounded"`, `"Session"`, `"Eventually"`).

---

## 5. Unified `BaseVectorStore` Interface

All vector stores in PolyRAG adhere to the exact same contract:

```python
# 1. Clear database
vdb.clear()

# 2. Add pre-computed vectors and documents
vdb.add_documents(vectors=vectors, documents=documents, batch_size=5000)

# 3. Query nearest neighbors
results = vdb.search(query_vector=query_vector, top_k=5)
# Each result item:
# {
#     "score": 0.89,            # Cosine similarity or normalized score
#     "distance": 0.11,         # Distance metric
#     "document": {
#         "_id": "doc_1#0_a1b2c3d4",
#         "text": "Chunk content...",
#         "source": "manual.pdf",
#         "chunk_id": 0,
#         ...
#     }
# }

# 4. Count total items
total = vdb.count()

# 5. Sample records
sample = vdb.peek(limit=5)
```
