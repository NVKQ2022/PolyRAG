# API Reference: Services

The `polyrag.service` module provides developer-facing facades designed for zero-boilerplate integration.

---

## `RAGService`

```python
class RAGService:
    def __init__(
        self,
        client: Any = None,
        chunking_service: BaseChunker | None = None,
        embedding_service: BaseEmbeddingModel | None = None,
        vector_db: BaseVectorStore | None = None,
        model_name: str | None = None,
    ) -> None: ...
```

### Factory Constructors

#### `RAGService.from_env(...)`
Reads environment variables (or `.env`) and initializes matching adapters:
```python
@classmethod
def from_env(
    cls,
    persist_dir: str | Path | None = "./chroma_db",
    collection_name: str = "rfc_documents",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> RAGService: ...
```

#### `RAGService.create(...)`
Convenience factory constructor with sensible defaults:
```python
@classmethod
def create(
    cls,
    model_name: str = "gpt-4o-mini",
    embedding_model: str = "all-MiniLM-L6-v2",
    persist_dir: str | Path | None = None,
    collection_name: str = "documents",
) -> RAGService: ...
```

### Ingestion Methods

#### `service.ingest(text, source="document", metadata=None)`
Splits, embeds, and indexes document text into the vector database.
- **Returns**: `list[dict[str, Any]]` of indexed document chunks.

#### `service.ingest_file(file_path, metadata=None)`
Reads a local file (`.txt`, `.md`, etc.) and indexes it.

#### `service.ingest_directory(dir_path, glob_pattern="*.txt", metadata=None)`
Recursively scans and indexes all matching files within a directory.

### Query Methods

#### `service.retrieve(query, top_k=5)`
Computes the query embedding and performs nearest-neighbor vector search.
- **Returns**: `list[dict[str, Any]]` sorted by similarity score.

#### `service.format_context(search_results)`
Formats retrieved results into a clean string block with source headers.

#### `service.query(question, top_k=5)`
Executes full Retrieve-then-Read pipeline.
- **Returns**: `RAGResponse`.

#### `service.as_agentic(top_k=3, max_rounds=2, verbose=False)`
Converts the `RAGService` into an `AgenticRAGService` instance sharing the same vector store and models.

---

## `AgenticRAGService`

```python
class AgenticRAGService:
    def __init__(
        self,
        rag_service: RAGService,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> None: ...
```

### Factory Constructors

#### `AgenticRAGService.from_env(...)`
Instantiates an Agentic RAG service reading configuration from environment variables:
```python
@classmethod
def from_env(
    cls,
    top_k: int = 3,
    max_rounds: int = 2,
    verbose: bool = False,
    persist_dir: str | Path | None = "./chroma_db",
    collection_name: str = "rfc_documents",
) -> AgenticRAGService: ...
```

### Methods

#### `agentic_service.query(question)`
Executes the full iterative reasoning workflow:
1. Planning & query rewrite
2. Multi-round retrieval & deduplication
3. Fused reflection & synthesis
- **Returns**: `RAGResponse` (includes `agent_log`, `confidence`, `reasoning_summary`).

#### `agentic_service.decide_retrieval(question)`
Inspects the question and returns planning decisions:
`{"retrieval_needed": bool, "query": str, "reason": str}`.

#### `agentic_service.direct_answer(question)`
Bypasses retrieval for conversational or polite questions.

#### `agentic_service.reflect_and_evaluate(question, context, round_number)`
Evaluates factual completeness and returns evidence sufficiency status.
