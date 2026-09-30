# Dependency Injection (DI) Guide 💉

PolyRAG features a built-in, lightweight **Dependency Injection (DI) Container** (`polyrag.Container`) acting as the Composition Root for all services, adapters, and pipelines.

---

## Why Dependency Injection in RAG?

1. **Testability & Mocking**: Swap real LLM endpoints or vector databases with test doubles in 1 line.
2. **Lifecycle Control**: Explicit control over Singletons (e.g. vector stores, embeddings) vs Transient factories (fresh instances per request).
3. **Decoupled Architecture**: Pipelines depend strictly on abstract interface contracts, not concrete classes.

---

## 1. Quick Example: Using `Container`

```python
from polyrag import (
    Container,
    BaseChunker,
    BaseEmbeddingModel,
    BaseVectorStore,
    BaseLLMClient,
    RecursiveCharacterChunker,
    SentenceTransformerEmbedding,
    InMemoryVectorStore,
    OpenAILLM,
    RAGService,
)

# 1. Create a container
container = Container()

# 2. Register dependencies (as singletons or factories)
container.register_instance(BaseChunker, RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50))
container.register_instance(BaseEmbeddingModel, SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2"))
container.register_instance(BaseVectorStore, InMemoryVectorStore())
container.register_instance(BaseLLMClient, OpenAILLM(model_name="gpt-4o-mini"))

# 3. Build services or pipelines directly
service = container.build_service()
# or: service = RAGService.from_container(container)

service.ingest("Technical document text...", source="doc.txt")
response = service.query("My question?")
print(response.answer)
```

---

## 2. Factory and Transient Scopes

### Singleton Registration (Default)
Singletons are instantiated once and cached:

```python
container.register_factory(BaseVectorStore, lambda: ChromaVectorStore(persist_path="./chroma_db"), singleton=True)

store1 = container.resolve(BaseVectorStore)
store2 = container.resolve(BaseVectorStore)
assert store1 is store2  # Same instance
```

### Transient Registration
Produces a fresh instance on every resolution:

```python
container.register_factory(BaseChunker, lambda: RecursiveCharacterChunker(), singleton=False)

chunker1 = container.resolve(BaseChunker)
chunker2 = container.resolve(BaseChunker)
assert chunker1 is not chunker2  # Distinct instances
```

---

## 3. Pre-Configured Factory Constructors

### From Environment Variables
```python
container = Container.from_env(
    persist_dir="./chroma_db",
    collection_name="knowledge_base",
    embedding_model="all-MiniLM-L6-v2",
)
```

### Convenience Factory
```python
container = Container.create(
    vector_store=InMemoryVectorStore(),
    llm_client=OpenAILLM(model_name="gpt-4o"),
)
```

---

## 4. Pipeline Builders

The `Container` can directly assemble configured pipeline instances with all dependencies automatically wired:

```python
# Build a Naive RAG pipeline
naive_pipeline = container.build_naive_rag()

# Build an Agentic RAG pipeline with reflection
agentic_pipeline = container.build_agentic_rag(top_k=3, max_rounds=2, verbose=True)

# Build a ReAct Agent
react_agent = container.build_react_agent(max_steps=4)

# Build unified Facade Services
service = container.build_service()
agentic_service = container.build_agentic_service(top_k=3)
```

---

## 5. Mocking in Unit Tests

With Dependency Injection, testing your RAG application without hitting OpenAI or cloud databases is trivial:

```python
class MockLLM(BaseLLMClient):
    @property
    def model_name(self): return "mock"
    def complete(self, prompt, **kwargs): return "Mocked answer"
    def complete_json(self, prompt, **kwargs): return {"retrieval_needed": False}
    def chat(self, messages, **kwargs): return "Mocked chat"

container = Container.create(
    vector_store=InMemoryVectorStore(),
    llm_client=MockLLM(),
)

service = container.build_service()
response = service.query("Test question")
assert response.answer == "Mocked answer"
```
