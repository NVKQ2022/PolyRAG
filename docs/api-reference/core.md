# API Reference: Core Models & Interfaces

The `polyrag.core` module defines the core domain entities, data transfer objects, abstract ports, and exceptions.

---

## 1. Domain Models (`polyrag.core.models`)

### `Document`
Represents an unsegmented source document:
```python
@dataclass
class Document:
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None
```

### `Chunk`
Represents an extracted passage from a document:
```python
@dataclass
class Chunk:
    source: str
    chunk_id: int | str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_uuid: str | None = None

    @property
    def identifier(self) -> str:
        """Returns '{source}#{chunk_id}' (e.g. 'rfc1035.txt#42')."""
```

### `SearchResult`
Represents a scored vector match from similarity search:
```python
@dataclass
class SearchResult:
    score: float           # Similarity score (e.g. Cosine Similarity [-1, 1])
    distance: float        # Distance metric (e.g. Euclidean / L2)
    document: dict[str, Any]
```

### `RAGResponse`
Standard output object returned by `BaseRAG`, `NaiveRAG`, `AdvancedRAG`, and `AgenticRAG`:
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
*Note: Supports both attribute access (`response.answer`) and dictionary indexing (`response["answer"]`).*

### `AgentStep`
A single reasoning step in a `ReActAgent` trajectory:
```python
@dataclass
class AgentStep:
    step_num: int
    thought: str
    action: str
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    chunks_retrieved: int = 0
    took_ms: int = 0
```

### `AgentResponse`
Comprehensive output returned by `ReActAgent`:
```python
@dataclass
class AgentResponse:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_evidence: list[dict[str, Any]]
    trajectory: list[AgentStep]
    reasoning_summary: str
    confidence: float
    total_steps: int
    took_ms: int
    llm_calls: int
```

---

## 2. Abstract Interfaces (`polyrag.core.interfaces`)

### `BaseChunker`
```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]: ...
```

### `BaseEmbeddingModel`
```python
class BaseEmbeddingModel(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed_text(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]: ...
```

### `BaseVectorStore`
```python
class BaseVectorStore(ABC):
    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def add_documents(self, vectors: list[list[float]], documents: list[dict[str, Any]], batch_size: int = 5000) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def peek(self, limit: int = 5) -> Any: ...
```

### `BaseLLMClient`
```python
class BaseLLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str: ...

    @abstractmethod
    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]: ...

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...
```

---

## 3. Exceptions (`polyrag.exceptions`)

- `RAGException`: Root base class for all library exceptions.
- `ConfigurationError`: Missing environment variables, invalid settings, or unresolvable DI container interfaces.
- `IngestionError`: Document reading, parsing, or chunking failures.
- `RetrievalError`: Vector database search or similarity computation failures.
- `LLMGenerationError`: LLM API network timeouts, quota limits, or invalid JSON syntax parsing.
