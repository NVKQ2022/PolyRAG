# API Reference: Services & DI Container

The `polyrag.service` and `polyrag.container` modules provide high-level facades and composition tools.

---

## 1. `RAGService` (`polyrag.service.RAGService`)

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
Instantiates from environment variables:
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

## 3. `Container` (`polyrag.container.Container`)

```python
class Container:
    def __init__(self) -> None: ...
```

### Registration & Resolution
- `register_instance(interface: type[T], instance: T) -> Container`
- `register_factory(interface: type[T], factory: Callable[..., T], singleton: bool = True) -> Container`
- `resolve(interface: type[T]) -> T`
- `is_registered(interface: type) -> bool`

### Pipeline Builders
- `build_naive_rag() -> NaiveRAG`
- `build_advanced_rag(top_k=5, num_expanded_queries=3, min_relevance_score=0.0, verbose=False) -> AdvancedRAG`
- `build_agentic_rag(top_k=3, max_rounds=2, verbose=False) -> AgenticRAG`
- `build_react_agent(max_steps=4, default_top_k=5, verbose=False) -> ReActAgent`
- `build_service() -> RAGService`
- `build_agentic_service(top_k=3, max_rounds=2, verbose=False) -> AgenticRAGService`

### Factory Constructors
- `Container.create(...) -> Container`
- `Container.from_env(...) -> Container`
