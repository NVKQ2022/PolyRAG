# Module: `polyrag.embeddings` 🧬

The `polyrag.embeddings` module converts text into dense, normalized vector representations suitable for similarity search in vector databases.

---

## 1. Available Embedding Adapters

| Adapter | Backend | Network Required | Default Model | Dimension |
| :--- | :--- | :--- | :--- | :--- |
| **`SentenceTransformerEmbedding`** | Local PyTorch / HuggingFace | ❌ No (runs 100% locally offline) | `all-MiniLM-L6-v2` | 384 |
| **`OpenAIEmbedding`** | OpenAI / Azure API | ✅ Yes | `text-embedding-3-small` | 1536 |

---

## 2. `SentenceTransformerEmbedding`

Runs state-of-the-art embedding models locally on CPU or GPU without sending data outside your infrastructure.

### Installation
```bash
pip install "polyrag[embeddings]"
```

### Usage
```python
from polyrag.embeddings import SentenceTransformerEmbedding

# Default model: 'all-MiniLM-L6-v2' (dim: 384)
embedder = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

print(f"Embedding dimension: {embedder.dim}")

# Single text
vector = embedder.embed_text("Semantic search with dense vectors")
print(f"Vector length: {len(vector)}")

# Batch processing
vectors = embedder.embed_batch(
    ["First sentence", "Second sentence", "Third sentence"],
    batch_size=64,
)
```

### Key Features
* **Auto Dimension Detection**: Safely detects embedding dimension via `get_embedding_dimension()` across `sentence-transformers` versions.
* **L2 Normalized**: Embeddings are pre-normalized (`normalize_embeddings=True`), allowing fast dot product / cosine similarity comparisons.

---

## 3. `OpenAIEmbedding`

High-throughput, cloud-managed embedding adapter supporting OpenAI and OpenAI-compatible endpoints (vLLM, Ollama, Azure OpenAI).

### Installation
```bash
pip install "polyrag[openai]"
```

### Usage
```python
from polyrag.embeddings import OpenAIEmbedding

embedder = OpenAIEmbedding(
    model_name="text-embedding-3-small",  # 1536 dims (or 'text-embedding-3-large' - 3072 dims)
    api_key="sk-...",                     # Optional if OPENAI_API_KEY is in environment
)

vector = embedder.embed_text("Query text for retrieval")
```

### Custom Endpoints & Azure OpenAI
You can point `OpenAIEmbedding` to self-hosted vLLM or local Ollama servers:

```python
embedder = OpenAIEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
```

---

## 4. Custom Embedding Adapter

To connect any custom embedding engine (e.g. Cohere, Bedrock, Vertex AI), implement `BaseEmbeddingModel`:

```python
from polyrag.core.interfaces import BaseEmbeddingModel

class CohereEmbedding(BaseEmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "embed-english-v3.0"):
        import cohere
        self.client = cohere.Client(api_key=api_key)
        self.model_name = model_name

    @property
    def dim(self) -> int:
        return 1024

    def embed_text(self, text: str) -> list[float]:
        res = self.client.embed(texts=[text], model=self.model_name)
        return res.embeddings[0]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        res = self.client.embed(texts=texts, model=self.model_name)
        return res.embeddings
```
