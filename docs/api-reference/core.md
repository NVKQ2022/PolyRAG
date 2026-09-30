# API Reference: Core Models & Interfaces

The `polyrag.core` module defines the core domain entities, data transfer objects (DTOs), and abstract interface specifications.

---

## Data Models (`polyrag.core.models`)

### `Document`
Represents an input source document before chunking:
- `source: str`: Identifier or file name of the document.
- `text: str`: Raw content of the document.
- `metadata: dict[str, Any]`: Optional key-value metadata.
- `doc_id: str | None`: Optional unique document identifier.

### `Chunk`
A segmented passage extracted from a document:
- `source: str`: Source document name.
- `chunk_id: int | str`: Sequential or unique chunk identifier.
- `text: str`: Segmented passage text.
- `metadata: dict[str, Any]`: Preserved and chunk-specific metadata.
- `identifier: str` *(property)*: Returns `"{source}#{chunk_id}"` (e.g. `"rfc1035.txt#42"`).
- `to_dict() -> dict[str, Any]`: Serializes chunk into a dictionary.

### `SearchResult`
Represents a scored vector match from similarity search:
- `score: float`: Similarity score (higher = closer, typically cosine similarity `[-1, 1]`).
- `distance: float`: Vector distance.
- `document: dict[str, Any]`: The matched chunk record.
- `identifier: str` *(property)*: Canonical reference string.

### `RAGResponse`
Standard response object returned by RAG pipelines:
- `question: str`: The original user query.
- `answer: str`: The synthesized, grounded answer.
- `context: str`: Formatted context provided to the LLM.
- `sources: list[dict[str, Any]]`: List of cited document chunks.
- `confidence: float`: Estimated answer confidence `[0.0, 1.0]`.
- `reasoning_summary: str`: Brief summary of agent decisions.
- `agent_log: list[dict[str, Any]]`: Granular action-by-action execution records.
- `took_ms: int`: Total elapsed time in milliseconds.
- `llm_calls: int`: Number of LLM API completions invoked.
- *Supports dictionary indexing*: `response["answer"]` works identically to `response.answer`.

### `AgentStep`
A single step in a ReAct reasoning trajectory:
- `step_num: int`: Index of the step.
- `thought: str`: Internal reasoning about what information is missing.
- `action: str`: Tool name chosen.
- `action_input: dict[str, Any]`: Arguments passed to the tool.
- `observation: str`: Tool output fed back into the reasoning loop.
- `chunks_retrieved: int`: Count of newly discovered chunks.
- `took_ms: int`: Latency for this specific step.

### `AgentResponse`
Full output returned by `ReActAgent`:
- `question: str`: Original query.
- `answer: str`: Final synthesis with citations.
- `trajectory: list[AgentStep]`: Full chronological reasoning path.
- `sources: list[dict[str, Any]]`: Deduplicated cited references.
- `confidence: float`: Answer confidence.
- `took_ms: int`: Total latency.

---

## Abstract Interfaces (`polyrag.core.interfaces`)

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

## Exceptions (`polyrag.exceptions`)

All custom exceptions inherit from `RAGException`:
- `ConfigurationError`: Raised when environment variables or adapter settings are invalid.
- `IngestionError`: Raised during document reading, parsing, or chunking failures.
- `RetrievalError`: Raised during vector database search or similarity computation issues.
- `LLMGenerationError`: Raised on API timeout, quota exhaustion, or malformed JSON parsing.
