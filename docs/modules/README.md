# PolyRAG Modules Directory 📦

PolyRAG is architected into clean, decoupled, single-responsibility modules following hexagonal architecture (Ports and Adapters). Each module encapsulates a specific domain, strategy, or infrastructure provider.

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
                                     │ wires together
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
| **`polyrag.chunkers`** | Text segmenting and token boundary management | `FixedSizeChunker`, `RecursiveCharacterChunker` | [Chunkers Guide](chunkers.md) |
| **`polyrag.embeddings`** | Dense vector representations of text | `SentenceTransformerEmbedding`, `OpenAIEmbedding` | [Embeddings Guide](embeddings.md) |
| **`polyrag.vector_stores`**| Nearest-neighbor vector index and metadata storage | `InMemoryVectorStore`, `ChromaVectorStore`, `MilvusVectorStore` | [Overview](vector_stores/README.md) · [Chroma](vector_stores/chroma.md) · [Milvus](vector_stores/milvus.md) · [Memory](vector_stores/memory.md) |
| **`polyrag.llms`** | LLM completion, structured JSON parsing, and chat | `OpenAILLM` (supports OpenAI, Azure, and vLLM) | [LLMs Guide](llms.md) |
| **`polyrag.pipelines`** | End-to-end RAG workflows & reasoning paradigms | `BaseRAG`, `NaiveRAG`, `AdvancedRAG`, `AgenticRAG`, `ReActAgent` | [Pipelines Guide](pipelines.md) |
| **`polyrag.container`** | Dependency Injection container and composition root | `Container` | [Container Guide](container.md) |
| **`polyrag.app` / `service`** | High-level Application Context & pipeline factory | `PolyRAG`, `RAGService`, `AgenticRAGService` | [App & Service Guide](app.md) |

---

## 💡 Design Principles

1. **Dependency Inversion (DIP)**: High-level pipelines never depend directly on specific vendors (like Chroma, Milvus, or OpenAI). They depend strictly on the abstract interfaces defined in `polyrag.core.interfaces`.
2. **Pluggability**: Every component can be swapped in one line of code—whether via `PolyRAG.create(...)` or the `Container`.
3. **Graceful Fallbacks**: Optional external libraries (`chromadb`, `pymilvus`, `sentence-transformers`, `openai`) are imported lazily, providing clear error guidance if an extra is missing.
