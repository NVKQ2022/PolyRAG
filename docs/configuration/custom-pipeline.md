# Custom Pipeline Configuration

PolyRAG is architected around the **Dependency Inversion Principle**. Every component implements a well-defined abstract interface in `polyrag.core.interfaces`. You can configure and swap any piece of the pipeline without altering the core retrieval logic.

---

## 1. Configuring Chunkers

### `RecursiveCharacterChunker` (Recommended)
Splits text along natural structural boundaries (paragraphs, sentences, words):

```python
from polyrag import RecursiveCharacterChunker

chunker = RecursiveCharacterChunker(
    chunk_size=550,           # Target chunk length in characters
    chunk_overlap=50,         # Character overlap between adjacent chunks
    separators=["\n\n", "\n", ". ", " ", ""],
    drop_empty=True,          # Discard whitespace-only chunks
)
```

### `FixedSizeChunker`
Strict character window slicing with overlap:

```python
from polyrag import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=400,
    chunk_overlap=40,
    drop_empty=True,
)
```

---

## 2. Configuring Embeddings

### Local Offline Embeddings (`SentenceTransformerEmbedding`)
Runs entirely locally using PyTorch without external network calls:

```python
from polyrag import SentenceTransformerEmbedding

embedding_model = SentenceTransformerEmbedding(
    model_name="all-MiniLM-L6-v2",  # HuggingFace model identifier
    device="cuda",                   # 'cuda', 'cpu', or 'mps'
    normalize=True,                  # L2-normalize vectors for cosine similarity
)
```

### Remote Embeddings (`OpenAIEmbedding`)
Uses OpenAI's embedding API:

```python
from polyrag import OpenAIEmbedding

embedding_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    api_key="sk-...",
)
```

---

## 3. Configuring Vector Stores

### In-Memory Vector Store (`InMemoryVectorStore`)
Zero-dependency, thread-safe, cosine-similarity storage ideal for unit tests, rapid prototyping, and ephemeral workflows:

```python
from polyrag import InMemoryVectorStore

vector_store = InMemoryVectorStore()
```

### ChromaDB (`ChromaVectorStore`)
Persistent, disk-backed or remote collection vector storage:

```python
from polyrag import ChromaVectorStore

vector_store = ChromaVectorStore(
    persist_path="./chroma_db",       # Set None for in-memory Chroma
    collection_name="knowledge_base",
)
```

---

## 4. Configuring LLM Clients

### `OpenAILLM`
Supports both official OpenAI and Azure OpenAI endpoints:

```python
from polyrag import OpenAILLM

llm = OpenAILLM(
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
)
```

---

## 5. Implementing a Custom Component

You can implement your own adapters by subclassing the abstract interfaces in `polyrag.core`:

```python
from polyrag.core.interfaces import BaseEmbeddingModel

class CustomEmbedding(BaseEmbeddingModel):
    @property
    def dim(self) -> int:
        return 768

    def embed_text(self, text: str) -> list[float]:
        # Call your proprietary model or endpoint
        return [0.0] * 768

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]
```
