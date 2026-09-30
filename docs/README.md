# PolyRAG Documentation Index 📚

Welcome to the **PolyRAG** documentation. PolyRAG is a modular, multi-paradigm Retrieval-Augmented Generation (RAG) framework supporting Vector, Agentic, and Graph RAG architectures.

---

## 🗂️ Table of Contents

### 1. [Getting Started](getting-started/)
- [Installation Guide](getting-started/installation.md): Environment setup, minimal install, and optional dependency extras (`[openai]`, `[chroma]`, `[embeddings]`, `[all]`).
- [Quickstart Guide](getting-started/quickstart.md): 5-minute hands-on walkthrough with zero-API-key and end-to-end examples.

### 2. [Configuration](configuration/)
- [Environment Configuration](configuration/environment.md): Setting up `.env`, API keys, model parameters, and Azure OpenAI endpoints.
- [Custom Pipeline Configuration](configuration/custom-pipeline.md): Swapping chunkers, embedding models, vector stores, and implementing custom adapters.

### 3. [Architecture & Guides](guides/)
- [Naive RAG Guide](guides/naive-rag.md): The standard Retrieve-then-Read pipeline, optimal use cases, and limitations.
- [Agentic RAG Guide](guides/agentic-rag.md): Query rewriting, iterative multi-round retrieval, deduplication, and fused self-reflection.
- [ReAct Agent Guide](guides/react-agent.md): Autonomous Thought-Action-Observation reasoning loop with tool execution.

### 4. [API Reference](api-reference/)
- [Core Models & Interfaces](api-reference/core.md): Domain dataclasses (`Document`, `Chunk`, `SearchResult`, `RAGResponse`, `AgentResponse`), abstract interfaces, and exceptions.
- [Services Reference](api-reference/service.md): Complete method signatures for `RAGService` and `AgenticRAGService`.

### 5. [Roadmap](roadmap/)
- [GraphRAG & Hybrid Search](roadmap/graphrag.md): Architecture plans for knowledge graph extraction, community summarization, and Reciprocal Rank Fusion.
