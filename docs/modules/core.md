# Module: `polyrag.core` 🏛️

The `polyrag.core` module defines the domain entities, shared data transfer objects (DTOs), abstract ports (interfaces), and exception hierarchy across PolyRAG.

Starting in PolyRAG 0.1.5+, PolyRAG foundational primitives inherit directly from standard **LangChain** base classes, enabling zero-wrapper interoperability with the entire LangChain ecosystem while retaining PolyRAG's clean, high-performance interfaces.

---

## 1. Domain Entities (`polyrag.core.models`)

### `Document`
Inherits directly from `langchain_core.documents.Document` while exposing PolyRAG's ergonomic `.text`, `.source`, and `.doc_id` properties.

```python
from polyrag.core.models import Document

doc = Document(
    source="articles/kb_001.md",
    text="# Title\n\nFull raw content...",
    metadata={"author": "DevOps Team", "category": "Auth"},
    doc_id="doc_custom_123",  # Optional custom ID
)

# Standard LangChain properties
print(doc.page_content)  # "# Title\n\nFull raw content..."
print(doc.id)            # "doc_custom_123"

# PolyRAG properties
print(doc.text)          # "# Title\n\nFull raw content..."
print(doc.source)        # "articles/kb_001.md"
print(doc.doc_id)        # "doc_custom_123"
```

| Field / Property | Type | Description |
| :--- | :--- | :--- |
| `page_content` / `text` | `str` | Full raw text content. |
| `metadata` | `dict[str, Any]` | Key-value attributes (author, date, tags, source). |
| `id` / `doc_id` | `str \| None` | Unique document identifier. |
| `source` | `str` | Document source path or origin URI. |

---

### `Chunk`
Represents a discrete segmented section of a `Document` produced by a chunker.

```python
from polyrag.core.models import Chunk

chunk = Chunk(
    source="articles/kb_001.md",
    chunk_id=1,
    text="Symptoms:\n- Invalid token message",
    metadata={"tokens": 12},
)

# Canonical identifier: 'articles/kb_001.md#1'
print(chunk.identifier)
```

---

### Message Primitives
PolyRAG provides LangChain-compatible message models:
- `HumanMessage` (subclasses `langchain_core.messages.HumanMessage`)
- `AIMessage` (subclasses `langchain_core.messages.AIMessage`, supports `tool_calls`)
- `SystemMessage` (subclasses `langchain_core.messages.SystemMessage`)
- `ToolMessage` (subclasses `langchain_core.messages.ToolMessage`)

---

### `LangChainDocumentConverter`
Bidirectional converter between LangChain Document objects and PolyRAG Document models:
```python
from polyrag.core.models import LangChainDocumentConverter

# Convert a LangChain document to a PolyRAG document
poly_doc = LangChainDocumentConverter.to_polyrag_document(lc_doc)

# Convert a generator/list of LangChain documents
poly_docs = LangChainDocumentConverter.to_polyrag_documents(loader.lazy_load())
```

---

## 2. Abstract Ports & LangChain Bridges (`polyrag.core.interfaces`)

PolyRAG uses the **Dual Compatibility Bridge Pattern**: each PolyRAG interface subclasses its corresponding LangChain primitive and bidirectionally implements both method signatures.

```
┌─────────────────────────────────┐
│     LangChain Core Primitive    │  (VectorStore, Embeddings, TextSplitter, BaseChatModel)
└────────────────┬────────────────┘
                 │ (Inherits)
┌────────────────▼────────────────┐
│      PolyRAG Abstract Port      │  (BaseVectorStore, BaseEmbeddingModel, BaseChunker, BaseLLMClient)
└────────────────┬────────────────┘
                 │ (Implements)
┌────────────────▼────────────────┐
│ PolyRAG & LangChain Ecosystem   │  (InMemoryVectorStore, langchain_chroma.Chroma, langchain_milvus.Milvus, etc.)
└─────────────────────────────────┘
```

### `BaseVectorStore`
Subclasses `langchain_core.vectorstores.VectorStore`.

```python
from langchain_core.vectorstores import VectorStore
from polyrag.core.interfaces import BaseVectorStore

class BaseVectorStore(VectorStore, ABC):
    # LangChain Standard Methods
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]: ...
    def similarity_search_by_vector(self, embedding: list[float], k: int = 4, **kwargs: Any) -> list[Document]: ...
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]: ...
    def similarity_search_with_score_by_vector(self, embedding: list[float], k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]: ...

    # PolyRAG High-Performance Ports
    @abstractmethod
    def add_documents(self, documents: list[Any] | None = None, vectors: list[list[float]] | None = None, **kwargs: Any) -> list[str]: ...
    @abstractmethod
    def search(self, query_vector: list[float] | None = None, query: str = "", top_k: int = 5, **kwargs: Any) -> list[dict[str, Any]]: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def peek(self, limit: int = 5) -> Any: ...
    @abstractmethod
    def clear(self) -> None: ...
```

### `BaseEmbeddingModel`
Subclasses `langchain_core.embeddings.Embeddings`.

```python
from langchain_core.embeddings import Embeddings
from polyrag.core.interfaces import BaseEmbeddingModel

class BaseEmbeddingModel(Embeddings, ABC):
    # LangChain standard methods
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

    # PolyRAG ports
    def embed_text(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...
```

### `BaseChunker`
Subclasses `langchain_text_splitters.TextSplitter`.

```python
from langchain_text_splitters import TextSplitter
from polyrag.core.interfaces import BaseChunker

class BaseChunker(TextSplitter, ABC):
    # LangChain standard
    def split_text(self, text: str) -> list[str]: ...

    # PolyRAG port
    def chunk(self, text: str) -> list[str]: ...
```

### `BaseLLMClient`
Provides standard LLM generation and conforms to modern LangChain Runnable invocation (`invoke`, `stream`, `bind_tools`).

```python
class BaseLLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...
    def complete(self, prompt: str, **kwargs: Any) -> str: ...
    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]: ...
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...
    def invoke(self, input: Any, **kwargs: Any) -> AIMessage: ...
    def stream(self, input: Any, **kwargs: Any) -> Any: ...
    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any: ...
```

---

## 3. Exceptions (`polyrag.exceptions`)

All framework exceptions derive from `RAGException`:

* `ConfigurationError`: Raised when mandatory settings (e.g. API keys or unregistered DI interfaces) are missing.
* `IngestionError`: Raised during document reading or chunking errors.
* `RetrievalError`: Raised when vector store queries or searches fail.
* `LLMGenerationError`: Raised when model completion or chat calls fail.
