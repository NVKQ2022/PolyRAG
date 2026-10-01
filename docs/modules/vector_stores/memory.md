# InMemoryVectorStore (`polyrag.vector_stores.memory`)

The `InMemoryVectorStore` is a **zero-dependency, pure-Python** vector store that ships built directly into the core PolyRAG library. It performs exact nearest-neighbor search using cosine similarity in RAM.

---

## 1. When to Use

* **Unit Testing & CI/CD**: Run deterministic tests without provisioning databases, Docker containers, or SQLite lock files.
* **Rapid Prototyping**: Experiment with chunking and embeddings without configuring database paths.
* **Ephemeral Sessions**: Conversational chat sessions or small document question-answering where data does not need to persist across application restarts.
* **Air-Gapped / Minimal Environments**: Environments where binary packages (`chromadb`, `pymilvus`) cannot be installed.

---

## 2. Constructor

```python
from polyrag.vector_stores import InMemoryVectorStore

# Zero-dependency, all parameters optional
vdb = InMemoryVectorStore(
    metric="cosine",  # "cosine", "l2", "dot", or "ip"
)

# Or seed with initial data:
seeded_vdb = InMemoryVectorStore(
    metric="cosine",
    initial_documents=[{"_id": "1", "text": "Cached doc"}],
    initial_vectors=[[0.1, 0.2, 0.3]],
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `metric` | `str` | `"cosine"` | Similarity metric: `"cosine"`, `"l2"`, or `"dot"` / `"ip"`. |
| `initial_documents` | `list[dict] \| None` | `None` | Optional initial list of document dicts to seed the store. |
| `initial_vectors` | `list[list[float]] \| None` | `None` | Optional initial list of vectors matching `initial_documents`. |

---

## 3. Mathematical Mechanics

`InMemoryVectorStore` implements exact Cosine Similarity using pure Python math:

$$\text{similarity}(A, B) = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \cdot \sqrt{\sum_{i=1}^n B_i^2}}$$

* **Score**: $\text{score} = \text{similarity} \in [-1, 1]$
* **Distance**: $\text{distance} = 1.0 - \text{score}$
* **Zero-Vector Protection**: Safely returns $0.0$ if either vector has zero magnitude ($\|A\| = 0$).

---

## 4. Method Reference

### `add_documents(vectors, documents, batch_size=5000)`
Appends vectors and document dictionaries to internal memory lists:

```python
vectors = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
]

documents = [
    {"_id": "doc1", "text": "First chunk", "source": "guide.md"},
    {"_id": "doc2", "text": "Second chunk", "source": "faq.md"},
]

vdb.add_documents(vectors=vectors, documents=documents)
```

---

### `search(query_vector, top_k=5) -> list[dict[str, Any]]`
Computes cosine similarity against all stored vectors, sorts descending, and returns the top-$k$ matches:

```python
results = vdb.search([0.1, 0.2, 0.3], top_k=1)
print(results[0]["score"])              # e.g. 1.0 (exact match)
print(results[0]["document"]["text"])   # "First chunk"
```

---

### `count() -> int`
Returns the total number of documents in memory:

```python
print(vdb.count())  # 2
```

---

### `peek(limit=5) -> Any`
Returns text previews of the first $N$ items:

```python
print(vdb.peek(limit=2))
# {'documents': ['First chunk', 'Second chunk']}
```

---

### `clear() -> None`
Wipes all stored vectors and documents from memory:

```python
vdb.clear()
assert vdb.count() == 0
```

---

## 5. End-to-End Testing Example

Use `InMemoryVectorStore` in `pytest` suites to verify RAG pipelines instantly with zero I/O:

```python
from polyrag.app import PolyRAG
from polyrag.vector_stores import InMemoryVectorStore
from polyrag.core.interfaces import BaseEmbeddingModel, BaseLLMClient

class FastMockEmbedding(BaseEmbeddingModel):
    @property
    def dim(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

class FastMockLLM(BaseLLMClient):
    @property
    def model_name(self) -> str:
        return "mock-llm"

    def complete(self, prompt: str, **kwargs) -> str:
        return "Test response"

    def complete_json(self, prompt: str, **kwargs) -> dict:
        return {"answer": "Test response"}

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return "Test chat"

def test_rag_pipeline():
    vdb = InMemoryVectorStore()
    rag = PolyRAG.create(
        vector_store=vdb,
        embedding_model=FastMockEmbedding(),
        llm_client=FastMockLLM(),
    )

    rag.ingest_text("PolyRAG is a multi-paradigm RAG framework.", source="intro.txt")
    assert vdb.count() == 1

    response = rag.query("What is PolyRAG?")
    assert response.answer == "Test response"
    assert len(response.sources) == 1
```

---

## 6. Limitations

* **Non-Persistent**: Data is lost when the Python process exits. For persistent storage, use [`ChromaVectorStore`](chroma.md) or [`MilvusVectorStore`](milvus.md).
* **RAM Bound**: Because all vectors reside in Python memory, it is suited for small-to-medium collections ($< 100\text{k}$ vectors). For larger datasets, use indexed vector databases.
