# PolyRAG Documentation Index 📚

Welcome to the **PolyRAG** documentation. PolyRAG is a modular, multi-paradigm Retrieval-Augmented Generation (RAG) framework supporting Vector, Agentic, and Graph RAG architectures.

---

## 🗂️ Documentation Directory

### 1. [Getting Started](getting-started/)
- [Installation Guide](getting-started/installation.md): Environment requirements, minimal installation, and optional dependency extras (`[openai]`, `[chroma]`, `[embeddings]`, `[all]`).
- [Quickstart Guide](getting-started/quickstart.md): 5-minute hands-on walkthrough with zero-API-key and production examples.
- [Publishing Guide](getting-started/publishing.md): Complete instructions for building and publishing to TestPyPI and PyPI.

### 2. [Configuration](configuration/)
- [Environment Configuration](configuration/environment.md): Setting up `.env`, API keys, model parameters, and Azure OpenAI endpoints.
- [Custom Pipeline Configuration](configuration/custom-pipeline.md): Swapping chunkers, embedding models, vector stores, and implementing custom adapters.
- [Dependency Injection Guide](configuration/dependency-injection.md): Using the `Container` to register, resolve, and wire components and pipelines.

### 3. [Architecture & Guides](guides/)
- [The RAG Class Hierarchy](guides/class-hierarchy.md): Comprehensive architectural guide to the `BaseRAG` object-oriented hierarchy.
- [Naive RAG Guide](guides/naive-rag.md): The standard Retrieve-then-Read pipeline, optimal use cases, and limitations.
- [Advanced RAG Guide](guides/advanced-rag.md): Pre-retrieval query expansion, multi-query parallel search, and RRF re-ranking.
- [Agentic RAG Guide](guides/agentic-rag.md): Planning, semantic query rewriting, iterative multi-round retrieval, deduplication, and fused self-reflection.
- [ReAct Agent Guide](guides/react-agent.md): Autonomous Thought-Action-Observation reasoning loop with dynamic tool execution.

### 4. [Deep Agents & Advanced Architecture](deep-agents/)
- [Deep Agents in RAG](deep-agents/README.md): Comprehensive analysis of cost, complexity, failure modes, and the 8-step workflow for long-horizon research vs. standard RAG.

### 5. [Modules Documentation](modules/)
- [Modules Overview & Catalog](modules/README.md): Architecture overview and index of all PolyRAG packages.
- [Core Module](modules/core.md): Domain models (`Document`, `Chunk`, `SearchResult`, `RAGResponse`), abstract ports, and exceptions.
- [Chunkers Module](modules/chunkers.md): `FixedSizeChunker` and `RecursiveCharacterChunker` algorithms and configuration.
- [Embeddings Module](modules/embeddings.md): `SentenceTransformerEmbedding` (local offline) and `OpenAIEmbedding` (cloud).
- [Vector Stores Module](modules/vector_stores/README.md): [ChromaDB](modules/vector_stores/chroma.md), [Milvus](modules/vector_stores/milvus.md), and [InMemory](modules/vector_stores/memory.md).
- [LLMs Module](modules/llms.md): `OpenAILLM` with support for OpenAI, Azure OpenAI, and local inference engines.
- [Pipelines Module](modules/pipelines.md): `BaseRAG`, `NaiveRAG`, `AdvancedRAG`, `AgenticRAG`, and `ReActAgent`.
- [Container Module](modules/container.md): Inversion of Control (IoC) Dependency Injection `Container`.
- [App & Service Module](modules/app.md): Central `PolyRAG` application context and legacy `RAGService` compatibility layer.
- [Adapters & Ecosystem Bridges](modules/adapters/README.md): [LangChain Document Loader Bridge](modules/adapters/langchain.md).

### 6. [API Reference](api-reference/)
- [Core Models & Interfaces](api-reference/core.md): Domain dataclasses (`Document`, `Chunk`, `SearchResult`, `RAGResponse`, `AgentResponse`), abstract interfaces, and exceptions.
- [Pipelines Reference](api-reference/pipelines.md): Complete method signatures for `BaseRAG`, `NaiveRAG`, `AdvancedRAG`, `AgenticRAG`, and `ReActAgent`.
- [Services & DI Container](api-reference/service.md): Complete method signatures for `RAGService`, `AgenticRAGService`, and `Container`.

### 7. [Roadmap](roadmap/)
- [GraphRAG & Hybrid Search](roadmap/graphrag.md): Architecture plans for knowledge graph extraction, community summarization, and Reciprocal Rank Fusion.
