# PolyRAG 🧬

[![PyPI Version](https://img.shields.io/pypi/v/polyrag.svg)](https://pypi.org/project/polyrag/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/polyrag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PolyRAG** is a modular, multi-paradigm Retrieval-Augmented Generation (RAG) framework designed to unify multiple RAG architectures—from standard **Naive 1-Shot RAG** and **Advanced Multi-Query RAG**, to autonomous **Agentic RAG** and future expansions like **GraphRAG**.

---

## 🏛️ The RAG Class Hierarchy

PolyRAG is architected around an object-oriented class hierarchy rooted in `BaseRAG`:

```
                       ┌─────────────────────────┐
                       │         BaseRAG         │
                       │  (Abstract Base Class)  │
                       └────────────┬────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
  │   NaiveRAG   │           │ AdvancedRAG  │           │  AgenticRAG  │
  │ (1-Shot RAG) │           │ (Expand/RRF) │           │ (Reflection) │
  └──────────────┘           └──────────────┘           └──────┬───────┘
                                                               │
                                                               ▼
                                                        ┌──────────────┐
                                                        │  ReActAgent  │
                                                        │ (ReActRAG)   │
                                                        └──────────────┘
```

All derived pipelines inherit shared document ingestion (`ingest_text`, `ingest_file`, `ingest_directory`), vector search (`retrieve`), and context assembly (`format_context`) from `BaseRAG`.

---

## 🌟 Key Features

- **Multi-Paradigm Pipelines**:
  - `NaiveRAG`: Standard retrieve-then-read pipeline with minimal latency.
  - `AdvancedRAG`: Multi-query expansion, parallel search, and Reciprocal Rank Fusion (RRF) re-ranking.
  - `AgenticRAG`: Planning, keyword rewriting, iterative multi-round search, deduplication, and fused self-reflection with source citations.
  - `ReActAgent`: Structured Thought-Action-Observation reasoning loop with dynamic tool execution.
  - *Coming Soon*: `GraphRAG` (knowledge graph entity extraction, community detection, and summarization).

- **Dependency Injection**:
  - Built-in `Container` acting as the Composition Root for all services, adapters, and pipelines.
  - Full support for Singleton and Transient scopes, lifecycle management, and 1-line test mocking.

- **Clean & Swappable Adapters**:
  - **Chunkers**: `FixedSizeChunker`, `RecursiveCharacterChunker` (boundary-aware).
  - **Embeddings**: `OpenAIEmbedding`, `SentenceTransformerEmbedding` (PyTorch).
  - **Vector Stores**: `ChromaVectorStore` (persistent / remote), `InMemoryVectorStore` (zero-dependency cosine similarity).
  - **LLMs**: `OpenAILLM` (supporting OpenAI chat completions and Azure OpenAI `responses.create`).

---

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Getting Started](docs/getting-started/)**: [Installation](docs/getting-started/installation.md) & [Quickstart](docs/getting-started/quickstart.md)
- **[Configuration](docs/configuration/)**: [Environment Variables](docs/configuration/environment.md), [Custom Pipelines](docs/configuration/custom-pipeline.md), & [Dependency Injection Guide](docs/configuration/dependency-injection.md)
- **[Guides](docs/guides/)**: [Class Hierarchy](docs/guides/class-hierarchy.md), [Naive RAG](docs/guides/naive-rag.md), [Advanced RAG](docs/guides/advanced-rag.md), [Agentic RAG](docs/guides/agentic-rag.md), & [ReAct Agent](docs/guides/react-agent.md)
- **[API Reference](docs/api-reference/)**: [Core Entities](docs/api-reference/core.md), [Pipelines](docs/api-reference/pipelines.md), & [Services / Container](docs/api-reference/service.md)
- **[Roadmap](docs/roadmap/)**: [GraphRAG & Hybrid Search](docs/roadmap/graphrag.md)

---

## 📦 Installation

```bash
# Minimal installation (core dataclasses, interfaces, in-memory store)
pip install polyrag

# With OpenAI support
pip install "polyrag[openai]"

# With local HuggingFace embeddings
pip install "polyrag[embeddings]"

# With ChromaDB vector store
pip install "polyrag[chroma]"

# Complete installation with all adapters
pip install "polyrag[all]"
```

---

## 🚀 Quick Start

### 1. Instant Setup from Environment

```python
from polyrag import RAGService

# Automatically loads configuration from environment variables (.env)
service = RAGService.from_env()

# Ingest raw text, a file, or an entire folder
service.ingest("RFC 1035 specifies DNS domain name syntax.", source="rfc1035.txt")
service.ingest_file("data/rfc9000.txt")
service.ingest_directory("docs/", glob_pattern="*.txt")

# End-to-end question answering
response = service.query("How does DNS translate domain names?")
print("Answer:", response.answer)
print("Sources:", response.sources)
```

---

### 2. Upgrading Through the Pipeline Hierarchy

Upgrade your pipeline strategy in 1 line as question complexity increases:

```python
# Level 1: Standard 1-Shot RAG
naive_res = service.query("What is DNS?")

# Level 2: Advanced RAG (Multi-Query Expansion & RRF Re-ranking)
advanced = service.create_advanced_rag(num_expanded_queries=3, top_k=5)
advanced_res = advanced.query("How does DNS handle packet truncation?")

# Level 3: Agentic RAG (Autonomous Multi-Round Loop & Fused Reflection)
agentic = service.create_agentic_rag(max_rounds=2, top_k=3, verbose=True)
agentic_res = agentic.query(
    "What transport protocol does HTTP/3 rely on, and how does connection migration work?"
)

print("Agentic Grounded Answer:\n", agentic_res.answer)
print("Confidence:", agentic_res.confidence)
print("Trajectory:", agentic_res.reasoning_summary)
```

---

### 3. Using Dependency Injection

```python
from polyrag import Container

# Build pre-wired container from environment
container = Container.from_env()

# Resolve any pipeline directly
pipeline = container.build_agentic_rag(top_k=3, max_rounds=2)
response = pipeline.query("Explain TLS 1.3 0-RTT handshakes.")
print(response.answer)
```

---

## 🧪 Testing

PolyRAG includes a complete suite of unit tests verifying all chunkers, stores, pipelines, and the DI container:

```bash
pytest tests
```

---

## 🛠️ Building & Publishing

To package PolyRAG for distribution:

```bash
# Build wheel and sdist
python -m build

# Upload to PyPI using twine
twine upload dist/*
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
