# PolyRAG Modules Directory 📦

PolyRAG is architected into clean, decoupled, single-responsibility modules built directly on top of native LangChain primitives.

---

## 🗺️ Module Architecture & Overview

```
                                ┌─────────────────────────┐
                                │   polyrag.app (PolyRAG)  │
                                └────────────┬────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
    │ polyrag.container│             │polyrag.pipelines│             │  polyrag.core   │
    │   (DI Engine)   │             │ (RAG Strategies)│             │ (Models/Ports)  │
    └────────┬────────┘             └────────┬────────┘             └────────┬────────┘
             │                               │                               │
             └───────────────────────┬───────┴───────────────────────────────┘
                                     │ orchestrates
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │polyrag.chunkers │     │polyrag.embeddings│     │polyrag.vector_stores│
    │ (Text Splitting)│     │(Vector Encoders)│     │(Storage/Search) │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  polyrag.llms   │
                            │(Generation/Chat)│
                            └─────────────────┘
```

---

## 📚 Module Catalog

| Module | Purpose | Key Classes & Exports | Documentation |
| :--- | :--- | :--- | :--- |
| **`polyrag.core`** | Domain models, abstract ports, and exceptions | `Document`, `Chunk`, `SearchResult`, `RAGResponse`, `BaseChunker`, `BaseEmbeddingModel`, `BaseVectorStore`, `BaseLLMClient` | [Core Guide](core.md) |
| **`polyrag.chunkers`** | Text segmenting and token boundary management | `FixedSizeChunker`, `RecursiveCharacterChunker` (subclassing LangChain's `TextSplitter`) | [Chunkers Guide](chunkers.md) |
| **`polyrag.embeddings`** | Dense vector representations of text | LangChain `Embeddings`, `FakeEmbeddings`, `resolve_embedding_model` | [Embeddings Guide](embeddings.md) |
| **`polyrag.vector_stores`**| Nearest-neighbor vector index and metadata storage | `InMemoryVectorStore`, LangChain `VectorStore` (`Chroma`, `Milvus`, etc.), `resolve_vector_store` | [Vector Stores Guide](vector-stores.md) |
| **`polyrag.llms`** | LLM completion, structured JSON parsing, and chat | LangChain `BaseChatModel`, `ChatOpenAI`, `resolve_llm_client` | [LLMs Guide](llms.md) |
| **`polyrag.pipelines`** | End-to-end RAG workflows & reasoning paradigms | `BaseRAG`, `NaiveRAG`, `AdvancedRAG`, `AgenticRAG`, `ReActAgent` | [Pipelines Guide](pipelines.md) |
| **`polyrag.container`** | Dependency Injection container and composition root | `Container` | [Container Guide](container.md) |
| **`polyrag.app` / `service`** | High-level Application Context & pipeline factory | `PolyRAG`, `RAGService`, `AgenticRAGService` | [App & Service Guide](app.md) |

---

## 💡 Design Principles

1. **Native LangChain Interoperability**: PolyRAG leverages standard LangChain primitives (`VectorStore`, `Embeddings`, `BaseChatModel`, `TextSplitter`) directly—meaning any third-party LangChain integration works out-of-the-box.
2. **Zero-Dependency Local Dev**: Start prototyping instantly with built-in `InMemoryVectorStore` and `FakeEmbeddings` with zero required external API keys or vector services.
3. **Pluggability**: Every component can be swapped in one line of code—whether via `PolyRAG.create(...)` or the `Container`.
