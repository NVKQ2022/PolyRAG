# Module: `polyrag.core` 🏛️

The `polyrag.core` module defines the domain entities, shared data transfer objects (DTOs), abstract ports (interfaces), and exception hierarchy across PolyRAG.

---

## 1. Domain Entities (`polyrag.core.models`)

### `Document`
Represents an unsegmented source document before chunking.

```python
from polyrag.core.models import Document

doc = Document(
    source="articles/kb_001.md",
    text="# Title\n\nFull raw content...",
    metadata={"author": "DevOps Team", "category": "Auth"},
    doc_id="doc_custom_123",  # Optional custom ID
)
```

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `str` | *Required* | File path, URI, or identifier of document origin. |
| `text` | `str` | *Required* | Full raw text content. |
| `metadata` | `dict[str, Any]` | `{}` | Key-value attributes (author, date, tags). |
| `doc_id` | `str \| None` | `None` | Optional unique identifier. |

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

| Property / Field | Type | Description |
| :--- | :--- | :--- |
| `source` | `str` | Document source from which the chunk originated. |
| `chunk_id` | `int \| str` | Zero-indexed or sequential section counter. |
| `text` | `str` | Text content of the chunk. |
| `metadata` | `dict[str, Any]` | Chunk-specific metadata. |
| `identifier` | `str` *(property)* | Canonical identifier formatted as `{source}#{chunk_id}`. |
| `to_dict()` | `dict[str, Any]` | Converts chunk to a flat dictionary for vector database storage. |

---

### `SearchResult`
Represents a nearest-neighbor vector match returned by a vector store.

```python
@dataclass
class SearchResult:
    score: float           # Similarity score (e.g. Cosine Similarity [-1, 1])
    distance: float        # Distance metric (e.g. L2 or 1 - Cosine)
    document: dict[str, Any] # Contains _id, text, source, and metadata
```

---

### `RAGResponse`
Standard output model returned by all RAG pipelines (`NaiveRAG`, `AdvancedRAG`, `AgenticRAG`, `PolyRAG.query()`).

```python
@dataclass
class RAGResponse:
    question: str
    answer: str
    context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    took_ms: int = 0
    confidence: float = 1.0
    reasoning_summary: str = ""
    agent_log: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 1
```

---

### `AgentResponse` & `AgentStep`
Output produced by the autonomous [`ReActAgent`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/pipelines/react.py):

* **`AgentStep`**: Records a single `thought`, `action` (tool name), `action_input`, and tool `observation`.
* **`AgentResponse`**: Contains the final `question`, `answer`, full `steps: list[AgentStep]`, `tool_calls_count`, and elapsed `took_ms`.

---

## 2. Abstract Ports (`polyrag.core.interfaces`)

PolyRAG uses the Ports & Adapters pattern. Custom infrastructure can be introduced by subclassing any of these interfaces:

### `BaseChunker`
```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split input document text into discrete chunks."""
        raise NotImplementedError
```

### `BaseEmbeddingModel`
```python
class BaseEmbeddingModel(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """Return dimensionality of the embedding vector."""
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        """Generate normalized embeddings for multiple texts."""
        raise NotImplementedError
```

### `BaseVectorStore`
```python
class BaseVectorStore(ABC):
    @abstractmethod
    def clear(self) -> None:
        """Clear all records from the vector store."""
        raise NotImplementedError

    @abstractmethod
    def add_documents(
        self,
        vectors: list[list[float]],
        documents: list[dict[str, Any]],
        batch_size: int = 5000,
    ) -> None:
        """Add pre-computed vectors and document records."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform nearest-neighbor search for a query embedding vector."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return total document count in the vector collection."""
        raise NotImplementedError

    @abstractmethod
    def peek(self, limit: int = 5) -> Any:
        """Preview sample records from the vector store."""
        raise NotImplementedError
```

### `BaseLLMClient`
```python
class BaseLLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError
```

---

## 3. Exceptions (`polyrag.exceptions`)

All framework exceptions derive from `PolyRAGError`:

* `ConfigurationError`: Raised when mandatory settings (e.g. API keys or directories) are missing.
* `RetrievalError`: Raised when vector store queries fail.
* `LLMGenerationError`: Raised when the LLM returns API errors or fails to generate text.
* `InvalidJSONError`: Raised when structured JSON generation cannot be parsed.
