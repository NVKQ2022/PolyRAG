# API Reference: Pipelines

The `polyrag.pipelines` module contains the RAG architecture hierarchy rooted in `BaseRAG`.

---

## 1. `BaseRAG` (`polyrag.pipelines.base`)

Abstract base class for all RAG architectures:

```python
class BaseRAG(ABC):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        llm_client: BaseLLMClient | None = None,
        chunker: BaseChunker | None = None,
    ) -> None: ...
```

### Methods

#### `ingest_text(text: str, source: str = "document", metadata: dict | None = None) -> list[dict]`
Segments text using `self.chunker`, computes embeddings via `self.embedding_model`, and stores them in `self.vector_store`.

#### `ingest_file(file_path: Path | str, metadata: dict | None = None) -> list[dict]`
Reads a local text/markdown file and ingests it.

#### `ingest_directory(dir_path: Path | str, glob_pattern: str = "*.txt", metadata: dict | None = None) -> list[dict]`
Recursively scans and indexes all matching files in a directory.

#### `retrieve(query: str, top_k: int = 5) -> list[dict]`
Generates embedding for `query` and searches the top-k nearest chunks in `self.vector_store`.

#### `format_context(search_results: list[dict]) -> str`
Formats retrieved search results into clean context blocks with source headers.

#### `execute(question: str, **kwargs: Any) -> RAGResponse | AgentResponse`
*(Abstract)* Subclasses must implement this method to define their answering workflow.

#### `query(question: str, **kwargs: Any) -> RAGResponse | AgentResponse`
Convenience alias for `execute()`.

---

## 2. `NaiveRAG` (`polyrag.pipelines.naive`)

Inherits from `BaseRAG`. Implements standard 1-shot Retrieve-then-Read:

```python
class NaiveRAG(BaseRAG):
    def execute(
        self,
        question: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RAGResponse: ...
```

---

## 3. `AdvancedRAG` (`polyrag.pipelines.advanced`)

Inherits from `BaseRAG`. Implements pre-retrieval query expansion and post-retrieval Reciprocal Rank Fusion (RRF):

```python
class AdvancedRAG(BaseRAG):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        llm_client: BaseLLMClient,
        chunker: BaseChunker | None = None,
        top_k: int = 5,
        num_expanded_queries: int = 3,
        min_relevance_score: float = 0.0,
        verbose: bool = False,
    ) -> None: ...

    def expand_query(self, question: str) -> list[str]: ...
    def reciprocal_rank_fusion(self, search_runs: list[list[dict]], rrf_k: int = 60) -> list[dict]: ...
    def execute(
        self,
        question: str,
        top_k: int | None = None,
        expand_queries: bool = True,
        re_rank: bool = True,
        **kwargs: Any,
    ) -> RAGResponse: ...
```

---

## 4. `AgenticRAG` (`polyrag.pipelines.agentic`)

Inherits from `BaseRAG`. Implements planning, iterative multi-round retrieval, deduplication, and fused reflection:

```python
class AgenticRAG(BaseRAG):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        llm_client: BaseLLMClient,
        chunker: BaseChunker | None = None,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
    ) -> None: ...

    def decide_retrieval(self, question: str) -> dict[str, Any]: ...
    def direct_answer(self, question: str) -> str: ...
    def reflect_and_evaluate(self, question: str, context: str, round_number: int, max_rounds: int) -> dict[str, Any]: ...
    def execute(
        self,
        question: str,
        top_k: int | None = None,
        max_rounds: int | None = None,
    ) -> RAGResponse: ...
```

---

## 5. `ReActAgent` / `ReActRAG` (`polyrag.pipelines.react`)

Inherits from `BaseRAG`. Implements Thought-Action-Observation loop with tool invocation:

```python
class ReActAgent(BaseRAG):
    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        chunker: BaseChunker | None = None,
        max_steps: int = 4,
        default_top_k: int = 5,
        verbose: bool = False,
    ) -> None: ...

    def execute(
        self,
        question: str,
        top_k: int | None = None,
        max_steps: int | None = None,
    ) -> AgentResponse: ...
```
