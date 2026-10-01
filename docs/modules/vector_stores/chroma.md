# ChromaVectorStore (`polyrag.vector_stores.chroma`)

The `ChromaVectorStore` provides persistent local vector storage backed by **ChromaDB**. It uses SQLite for metadata persistence and HNSW (Hierarchical Navigable Small World) graphs for nearest-neighbor vector indexing.

---

## 1. Installation

Install PolyRAG with the `[chroma]` optional extra:

```bash
pip install "polyrag[chroma]"
```

Or install `chromadb` directly:

```bash
pip install chromadb>=0.4.0
```

---

## 2. Constructor & Parameters

```python
from polyrag.vector_stores import ChromaVectorStore

vdb = ChromaVectorStore(
    persist_directory="chroma_db",
    collection_name="rfc_docs",
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `persist_directory` | `str` | `"chroma_db"` | Directory path on disk where SQLite metadata and vector index files will be stored. Created automatically if it does not exist. |
| `collection_name` | `str` | `"rfc_docs"` | Target collection name within ChromaDB. |

---

## 3. Storage & Indexing Mechanics

* **Persistence Engine**: Uses `chromadb.PersistentClient(path=persist_directory)`. Data survives application restarts.
* **Distance Metric**: Collections are initialized with `metadata={"hnsw:space": "cosine"}`. Chroma computes cosine distance:
  $$\text{distance} \in [0, 2]$$
  PolyRAG converts this to similarity score:
  $$\text{score} = 1.0 - \text{distance}$$
* **Deterministic ID Generation**: When inserting documents, IDs are automatically generated following the format:
  `{source_stem}_{chunk_id}_{uuid_hex_8}`
  *(e.g., `rfc1035_0_a1b2c3d4`)*. If a document already provides an `_id` field, that ID is preserved.
* **Telemetry**: Anonymized telemetry is disabled by default (`anonymized_telemetry=False`).

---

## 4. Method Reference

### `add_documents(vectors, documents, batch_size=5000)`
Adds pre-computed embeddings and document dictionaries to the collection in batches.

```python
vectors = [
    [0.12, 0.45, 0.78],
    [0.89, 0.23, 0.05],
]

documents = [
    {
        "text": "DNS translates domain names into IP addresses.",
        "source": "rfc1035.txt",
        "chunk_id": 0,
        "author": "Mockapetris",
    },
    {
        "text": "HTTP/2 introduces stream multiplexing over a single TCP connection.",
        "source": "rfc7540.txt",
        "chunk_id": 1,
        "category": "Web Protocols",
    },
]

vdb.add_documents(vectors=vectors, documents=documents, batch_size=1000)
```

* **Validation**: Raises `ValueError` if `len(vectors) != len(documents)` or `batch_size <= 0`.
* **Metadata Splitting**: The `"text"` key is stored in Chroma's `documents` field, while all other key-value pairs are stored in `metadatas`.

---

### `search(query_vector, top_k=5) -> list[dict[str, Any]]`
Searches nearest neighbors by cosine similarity.

```python
query_vec = [0.10, 0.40, 0.75]
results = vdb.search(query_vector=query_vec, top_k=2)

for hit in results:
    print(f"Score: {hit['score']:.4f} | Distance: {hit['distance']:.4f}")
    print(f"ID:    {hit['document']['_id']}")
    print(f"Text:  {hit['document']['text']}")
    print(f"Meta:  {hit['document'].get('source')}")
```

#### Output Structure
```python
[
    {
        "score": 0.9421,
        "distance": 0.0579,
        "document": {
            "_id": "rfc1035_0_9fa1b2c3",
            "text": "DNS translates domain names into IP addresses.",
            "source": "rfc1035.txt",
            "chunk_id": 0,
            "author": "Mockapetris",
        },
    }
]
```

---

### `count() -> int`
Returns total number of items currently stored in the collection:

```python
print(f"Total indexed chunks: {vdb.count()}")
```

---

### `peek(limit=5) -> Any`
Returns a sample of the first $N$ records directly from Chroma:

```python
sample = vdb.peek(limit=3)
print(sample["ids"])
print(sample["documents"])
```

---

### `clear() -> None`
Deletes and recreates the Chroma collection, wiping all records:

```python
vdb.clear()
assert vdb.count() == 0
```

---

## 5. End-to-End Usage in PolyRAG

```python
from polyrag.app import PolyRAG
from polyrag.vector_stores import ChromaVectorStore
from polyrag.embeddings import SentenceTransformerEmbedding
from polyrag.llms import OpenAILLM

# 1. Initialize persistent Chroma vector store
vdb = ChromaVectorStore(
    persist_directory="./production_data/chroma",
    collection_name="it_knowledge_base",
)

# 2. Build the unified PolyRAG pipeline
rag = PolyRAG.create(
    vector_store=vdb,
    embedding_model=SentenceTransformerEmbedding("all-MiniLM-L6-v2"),
    llm_client=OpenAILLM(model_name="gpt-4o-mini"),
)

# 3. Ingest documents (only needed once - persisted across runs)
if vdb.count() == 0:
    rag.ingest_directory("docs/", glob_pattern="**/*.md")

# 4. Query
response = rag.query("How do I configure OAuth 2.0 PKCE?")
print(response.answer)
print(f"Retrieved from: {[s.get('source') for s in response.sources]}")
```

---

## 6. Docker & Production Tips

* **Volume Mount**: When deploying inside Docker containers, mount `persist_directory` to a host volume to prevent data loss on container recreation:
  ```bash
  docker run -v /var/data/chroma:/app/chroma_db my-rag-service
  ```
* **Concurrent Writes**: Chroma SQLite uses file locks. If running multiple worker processes (e.g. `gunicorn -w 4`), ensure only one worker writes at a time, or switch to [`MilvusVectorStore`](milvus.md) for distributed high-concurrency writes.
