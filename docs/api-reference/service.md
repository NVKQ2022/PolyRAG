# API Reference: PolyRAG, Services & DI Container

The `polyrag.app`, `polyrag.service`, and `polyrag.container` modules provide the application setup context, service facades, and dependency injection tools.

---

## 1. `PolyRAG` (`polyrag.app.PolyRAG` / `polyrag.PolyRAG`)

The central application context and pipeline factory. It sets up components, manages shared document ingestion, and manufactures RAG pipelines.

```python
class PolyRAG:
    def __init__(
        self,
        chunker: BaseChunker | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        vector_store: BaseVectorStore | None = None,
        llm_client: BaseLLMClient | None = None,
        container: Container | None = None,
    ) -> None: ...
```

### Factory Constructors

#### `PolyRAG.from_env(...)`
Instantiates from environment variables (`OPENAI_API_KEY`, `MODEL_NAME`, Chroma paths):
```python
@classmethod
def from_env(
    cls,
    persist_dir: str | Path | None = "./chroma_db",
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> PolyRAG: ...
```

#### `PolyRAG.create(...)`
Convenience factory with sensible defaults:
```python
@classmethod
def create(
    cls,
    model_name: str = "gpt-4o-mini",
    embedding_model: str = "all-MiniLM-L6-v2",
    persist_dir: str | Path | None = None,
    collection_name: str = "documents",
    chunker: BaseChunker | None = None,
) -> PolyRAG: ...
```

#### `PolyRAG.from_container(container)`
Instantiates directly from an existing DI `Container`:
```python
@classmethod
def from_container(cls, container: Container) -> PolyRAG: ...
```

### Shared Ingestion Methods
- `ingest(text: str, source: str = "document", metadata: dict | None = None) -> list[dict]`
- `ingest_text(text: str, source: str = "document", metadata: dict | None = None) -> list[dict]`
- `ingest_file(file_path: Path | str, metadata: dict | None = None) -> list[dict]`
- `ingest_directory(dir_path: Path | str, glob_pattern: str = "*.txt", metadata: dict | None = None) -> list[dict]`

### Shared Retrieval & Context Formatting
- `retrieve(query: str, top_k: int = 5) -> list[dict]`
- `format_context(search_results: list[dict]) -> str`
- `query(question: str, top_k: int = 5) -> RAGResponse` *(Execution shortcut using default NaiveRAG)*

### Pipeline Factory Methods
- `create_naive_rag() -> NaiveRAG`
- `create_advanced_rag(top_k=5, num_expanded_queries=3, min_relevance_score=0.0, verbose=False) -> AdvancedRAG`
- `create_agentic_rag(top_k=3, max_rounds=2, verbose=False) -> AgenticRAG`
- `create_agentic_service(top_k=3, max_rounds=2, verbose=False) -> AgenticRAGService`
- `create_react_agent(max_steps=4, default_top_k=5, verbose=False) -> ReActAgent`

---

## 2. `RAGService` (`polyrag.service.RAGService`)

A high-level facade inheriting directly from `PolyRAG`. Retained for full backward compatibility:
- Re-exposes legacy parameter names (`client`, `chunking_service`, `embedding_service`, `vector_db`).
- Re-exposes legacy aliases (`as_advanced`, `as_agentic`).
- `create_agentic_rag()` on `RAGService` returns `AgenticRAGService`.

#### `RAGService.create(...)`
Convenience factory with sensible defaults:
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

#### `RAGService.from_container(container)`
Instantiates directly from an injected `Container`:
```python
@classmethod
def from_container(cls, container: Any) -> RAGService: ...
```

### Ingestion Methods
- `ingest(text: str, source: str = "document", metadata: dict | None = None) -> list[dict]`
- `ingest_file(file_path: Path | str, metadata: dict | None = None) -> list[dict]`
- `ingest_directory(dir_path: Path | str, glob_pattern: str = "*.txt", metadata: dict | None = None) -> list[dict]`

### Query Methods
- `retrieve(query: str, top_k: int = 5) -> list[dict]`
- `format_context(search_results: list[dict]) -> str`
- `query(question: str, top_k: int = 5) -> RAGResponse`

### Pipeline Creation & Factory Methods
- `create_advanced_rag(top_k=5, num_expanded_queries=3, min_relevance_score=0.0, verbose=False) -> AdvancedRAG`
- `create_agentic_rag(top_k=3, max_rounds=2, verbose=False) -> AgenticRAGService`
- `create_react_agent(max_steps=4, default_top_k=5, verbose=False) -> ReActAgent`
- *(Deprecated aliases: `as_advanced(...)`, `as_agentic(...)`)*

---

## 2. `AgenticRAGService` (`polyrag.service.AgenticRAGService`)

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
- `AgenticRAGService.from_env(top_k=3, max_rounds=2, verbose=False, ...) -> AgenticRAGService`
- `AgenticRAGService.from_container(container, top_k=3, max_rounds=2, verbose=False) -> AgenticRAGService`

### Methods
- `query(question: str) -> RAGResponse`
- `decide_retrieval(question: str) -> dict[str, Any]`
- `direct_answer(question: str) -> str`
- `reflect_and_evaluate(question: str, context: str, round_number: int) -> dict[str, Any]`

---

## 4. `Container` (`polyrag.container.Container`)

```python
class Container:
    def __init__(self) -> None: ...
```

### Registration & Resolution
- `register_instance(interface: type[T], instance: T) -> Container`
- `register_factory(interface: type[T], factory: Callable[..., T], singleton: bool = True) -> Container`
- `resolve(interface: type[T]) -> T`
- `is_registered(interface: type) -> bool`

### Pipeline & Application Builders
- `build_app() -> PolyRAG`
- `build_naive_rag() -> NaiveRAG`
- `build_advanced_rag(top_k=5, num_expanded_queries=3, min_relevance_score=0.0, verbose=False) -> AdvancedRAG`
- `build_agentic_rag(top_k=3, max_rounds=2, verbose=False) -> AgenticRAG`
- `build_react_agent(max_steps=4, default_top_k=5, verbose=False) -> ReActAgent`
- `build_service() -> RAGService`
- `build_agentic_service(top_k=3, max_rounds=2, verbose=False) -> AgenticRAGService`

### Factory Constructors
- `Container.create(...) -> Container`
- `Container.from_env(...) -> Container`
