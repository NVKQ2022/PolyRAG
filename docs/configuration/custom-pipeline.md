# Custom Pipeline Configuration

PolyRAG is architected around the **Dependency Inversion Principle**. Every component adheres to abstract interfaces defined in `polyrag.core.interfaces`. You can swap any component or write custom implementations without modifying core pipelines.

---

## 1. Chunkers (`BaseChunker`)

### `RecursiveCharacterChunker` (Recommended)
Recursively splits text using natural hierarchy (paragraphs, newlines, sentences, spaces):

```python
from polyrag import RecursiveCharacterChunker

chunker = RecursiveCharacterChunker(
    chunk_size=550,           # Max chunk character length
    chunk_overlap=50,         # Overlap between consecutive chunks
    separators=["\n\n", "\n", ". ", " ", ""],
    drop_empty=True,          # Discard whitespace chunks
)
```

### `FixedSizeChunker`
Fixed-window slicing with overlap:

```python
from polyrag import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=400,
    chunk_overlap=40,
    drop_empty=True,
)
```

---

## 2. Embeddings (`BaseEmbeddingModel`)

### `SentenceTransformerEmbedding` (Local PyTorch)
Runs completely offline on CPU, CUDA, or MPS:

```python
from polyrag import SentenceTransformerEmbedding

embedding_model = SentenceTransformerEmbedding(
    model_name="all-MiniLM-L6-v2",
    device="cuda",                   # 'cuda', 'cpu', 'mps'
    normalize=True,                  # L2-normalize vectors for cosine similarity
)
```

### `OpenAIEmbedding` (Cloud API)
```python
from polyrag import OpenAIEmbedding

embedding_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    api_key="sk-...",
)
```

---

## 3. Vector Stores (`BaseVectorStore`)

### `InMemoryVectorStore` (Zero-Dependency)
Thread-safe in-memory vector store with cosine similarity ranking. Perfect for testing and ephemeral tasks:

```python
from polyrag import InMemoryVectorStore

vector_store = InMemoryVectorStore()
```

### `ChromaVectorStore` (Persistent / Remote)
```python
from polyrag import ChromaVectorStore

vector_store = ChromaVectorStore(
    persist_path="./chroma_db",
    collection_name="production_docs",
)
```

---

## 4. LLM Clients (`BaseLLMClient`)

```python
from polyrag import OpenAILLM

llm = OpenAILLM(
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
)
```

---

## 5. Assembling Any Pipeline in the Hierarchy

Once components are configured, pass them to any pipeline deriving from `BaseRAG`:

```python
from polyrag import NaiveRAG, AdvancedRAG, AgenticRAG

# Standard Naive RAG
naive = NaiveRAG(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm_client=llm,
)

# Advanced RAG (Multi-Query Expansion & RRF)
advanced = AdvancedRAG(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm_client=llm,
    num_expanded_queries=3,
    top_k=5,
)

# Agentic RAG (Autonomous Planning & Fused Reflection)
agentic = AgenticRAG(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm_client=llm,
    top_k=3,
    max_rounds=2,
    verbose=True,
)
```

---

## 6. Implementing Custom Adapters

To integrate a new vector database (e.g. Qdrant, Pinecone) or embedding provider (e.g. Cohere, Gemini), subclass the base interfaces:

```python
from polyrag.core.interfaces import BaseVectorStore
from typing import Any

class CustomVectorStore(BaseVectorStore):
    def clear(self) -> None: ...
    def add_documents(self, vectors: list[list[float]], documents: list[dict[str, Any]], batch_size: int = 5000) -> None: ...
    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]: ...
    def count(self) -> int: ...
    def peek(self, limit: int = 5) -> Any: ...
```
