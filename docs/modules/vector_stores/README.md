# Module: `polyrag.vector_stores` 🗄️

The `polyrag.vector_stores` module provides vector database adapters that implement nearest-neighbor search, document indexing, and metadata persistence.

All stores implement the [`BaseVectorStore`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/core/interfaces.py#L40-L76) port, meaning you can swap backends without changing any application or pipeline code.

---

## 🧭 Specific Store Guides

| Store Guide | Backend Technology | Dependencies | Best For |
| :--- | :--- | :--- | :--- |
| **[`chroma.md`](chroma.md)** | ChromaDB | `chromadb>=0.4.0` | Embedded local file storage, lightweight Python apps, desktop utilities. |
| **[`milvus.md`](milvus.md)** | Milvus & Milvus Lite | `pymilvus>=2.4.0` | Production enterprise scale, Milvus Lite file, Docker Standalone, or Zilliz Cloud. |
| **[`memory.md`](memory.md)** | Pure Python In-Memory | *Zero dependencies* | Unit tests, CI/CD pipelines, ephemeral air-gapped sessions. |

---

## 📊 Comparison Matrix

| Feature | `InMemoryVectorStore` | `ChromaVectorStore` | `MilvusVectorStore` |
| :--- | :--- | :--- | :--- |
| **Persistence** | ❌ Ephemeral (RAM) | ✅ Local SQLite / HNSW | ✅ Local file, Server, or Cloud |
| **External Dependencies** | **None** (Built-in) | `chromadb` | `pymilvus` |
| **Embedded / Serverless** | ✅ Yes | ✅ Yes | ✅ Yes (via Milvus Lite) |
| **Distributed / Clustering**| ❌ No | ❌ No | ✅ Yes (Milvus Cluster) |
| **Managed Cloud Option** | ❌ No | ❌ No | ✅ Yes (Zilliz Cloud) |
| **Dynamic Schema ($meta)** | ✅ Python dict | ⚠️ Flat scalars | ✅ Native Dynamic Schema & JSON |
| **Supported Metrics** | Cosine | Cosine, L2, IP | COSINE, L2, IP |
| **Default Dimension** | Inferred dynamically | Inferred dynamically | Inferred dynamically |

---

## 🛠️ Unified Contract (`BaseVectorStore`)

Every store in this module guarantees the following public methods:

```python
from polyrag.core.interfaces import BaseVectorStore

class CustomVectorStore(BaseVectorStore):
    def clear(self) -> None: ...
    def add_documents(self, vectors: list[list[float]], documents: list[dict[str, Any]], batch_size: int = 5000) -> None: ...
    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]: ...
    def count(self) -> int: ...
    def peek(self, limit: int = 5) -> Any: ...
```
