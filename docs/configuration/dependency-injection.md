# Dependency Injection (DI) Container Guide 💉

PolyRAG includes a lightweight, thread-safe **Dependency Injection (DI) Container** (`polyrag.Container`). It serves as the **Composition Root** for resolving, configuring, and assembling all components and pipelines.

The `Container` supports both PolyRAG interfaces (`BaseVectorStore`, `BaseEmbeddingModel`, `BaseLLMClient`, `BaseChunker`) and native LangChain types (`VectorStore`, `Embeddings`, `BaseChatModel`, `TextSplitter`).

---

## 1. Core Concepts

| Concept | Description | Method |
| :--- | :--- | :--- |
| **Instance (Singleton)** | Registers an already instantiated object shared across all resolutions. | `container.register_instance(interface, instance)` |
| **Lazy Factory (Singleton)** | Calls a factory on first access and caches the created instance. | `container.register_factory(interface, factory, singleton=True)` |
| **Transient Factory** | Calls a factory callable on every resolution, returning a fresh instance. | `container.register_factory(interface, factory, singleton=False)` |
| **Resolution** | Type-safe lookup returning the registered implementation. | `container.resolve(interface)` |

---

## 2. Basic Setup & Usage

```python
from polyrag import (
    Container,
    BaseChunker,
    BaseEmbeddingModel,
    BaseVectorStore,
    BaseLLMClient,
    RecursiveCharacterChunker,
    InMemoryVectorStore,
    resolve_embedding_model,
    ChatOpenAI,
)

# 1. Initialize Container
container = Container()

# 2. Register Dependencies (supports both PolyRAG & LangChain interfaces)
container.register_instance(BaseChunker, RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50))
container.register_instance(BaseEmbeddingModel, resolve_embedding_model("fake", size=384))
container.register_instance(BaseVectorStore, InMemoryVectorStore())
container.register_instance(BaseLLMClient, ChatOpenAI(model="gpt-4o-mini"))

# 3. Resolve any component
store = container.resolve(BaseVectorStore)
assert isinstance(store, InMemoryVectorStore)
```

---

## 3. Pipeline Builders

The `Container` can directly instantiate fully wired pipelines adhering to the `BaseRAG` hierarchy:

```python
# Build a Naive RAG pipeline (1-shot)
naive = container.build_naive_rag()

# Build an Advanced RAG pipeline (Multi-Query Expansion + RRF Re-ranking)
advanced = container.build_advanced_rag(num_expanded_queries=3, top_k=5)

# Build an Agentic RAG pipeline (Autonomous Planning + Multi-Round Reflection)
agentic = container.build_agentic_rag(top_k=3, max_rounds=2, verbose=True)

# Build a ReAct Agent (Thought-Action-Observation tool loop)
react = container.build_react_agent(max_steps=4)

# Build central PolyRAG Application Context
app = container.build_app()
```

---

## 4. Pre-Wired Factory Constructors

### From Environment Variables
Reads `.env` settings for models, collections, and directories:
```python
container = Container.from_env(
    persist_dir="./chroma_db",
    collection_name="production_specs",
    embedding_model="all-MiniLM-L6-v2",
)
```

### With Custom Overrides
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vdb = Chroma(collection_name="docs", embedding_function=emb)

container = Container.create(
    vector_store=vdb,
    embedding_model=emb,
    chat_model=ChatOpenAI(model="gpt-4o-mini"),
)
```

---

## 5. Integrating with `PolyRAG`

You can pass a configured container directly to `PolyRAG`:

```python
from polyrag import PolyRAG

app = PolyRAG.from_container(container)
app.ingest_file("data/spec.txt")
response = app.query("How does connection migration work?")
print(response.answer)
```

---

## 6. Effortless Unit Testing & Mocking

Dependency Injection makes testing trivial by replacing real API calls and database connections with mock implementations:

```python
from polyrag import Container, BaseLLMClient, InMemoryVectorStore

class MockLLM(BaseLLMClient):
    @property
    def model_name(self): return "mock"
    def complete(self, prompt, **kwargs): return "Mocked response [doc#0]"
    def complete_json(self, prompt, **kwargs): return {"retrieval_needed": False}
    def chat(self, messages, **kwargs): return "Mocked chat"

# Wire test container
container = Container.create(
    vector_store=InMemoryVectorStore(),
    llm_client=MockLLM(),
)

# Test pipeline in isolation
pipeline = container.build_naive_rag()
pipeline.ingest_text("Test content", source="test.txt")
response = pipeline.query("Question?")

assert response.answer == "Mocked response [doc#0]"
```
