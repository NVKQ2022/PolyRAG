# Module: `polyrag.container` 💉

The `polyrag.container` module provides a lightweight, thread-safe Inversion of Control (IoC) / Dependency Injection container that serves as PolyRAG's **Composition Root**.

---

## 1. What is the `Container`?

The `Container` registers, constructs, and resolves all adapters and pipelines in your application. It decouples high-level pipelines from concrete third-party libraries (Chroma, Milvus, SentenceTransformers, OpenAI), enabling modular testing and configuration switching.

```
                    ┌───────────────────────────────┐
                    │      polyrag.Container        │
                    └───────────────┬───────────────┘
                                    │ resolves
     ┌──────────────────┬───────────┴───────────┬──────────────────┐
     ▼                  ▼                       ▼                  ▼
BaseChunker    BaseEmbeddingModel        BaseVectorStore      BaseLLMClient
```

---

## 2. Registering and Resolving Components

### Registering Pre-Built Instances
```python
from polyrag.container import Container
from polyrag.core.interfaces import BaseVectorStore, BaseEmbeddingModel
from polyrag.vector_stores import InMemoryVectorStore
from polyrag.embeddings import resolve_embedding_model

container = Container()

# Bind concrete instances to abstract interfaces
container.register_instance(BaseVectorStore, InMemoryVectorStore())
container.register_instance(BaseEmbeddingModel, resolve_embedding_model(None))

# Resolve at runtime
vdb = container.resolve(BaseVectorStore)
```

### Registering Lazy Factories
Factories are evaluated only when first requested. By default, factories act as singletons:

```python
from polyrag.core.interfaces import BaseLLMClient
from polyrag.llms import ChatOpenAI

# Singleton factory (only initialized on first .resolve() call)
container.register_factory(
    BaseLLMClient,
    lambda: ChatOpenAI(model="gpt-4o-mini"),
    singleton=True,
)
```

---

## 3. Pipeline Factory Methods

The `Container` knows how to wire and construct all RAG pipelines using its resolved dependencies:

```python
# Build pipelines directly from container configuration:
naive_rag = container.build_naive_rag()

advanced_rag = container.build_advanced_rag(
    top_k=5,
    num_expanded_queries=3,
    min_relevance_score=0.4,
)

agentic_rag = container.build_agentic_rag(
    top_k=3,
    max_rounds=2,
)

react_agent = container.build_react_agent(
    tools=[...],
    max_steps=5,
)

# Or build the unified PolyRAG application context
app = container.build_app()
```

---

## 4. `Container.from_env()`

Constructs a fully configured container automatically by reading environment variables:

```python
from polyrag.container import Container

# Inspects OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR, etc.
container = Container.from_env()

# Instantly ready for ingestion and retrieval
app = container.build_app()
```
