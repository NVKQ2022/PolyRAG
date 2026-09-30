# PolyRAG 🧬

[![PyPI Version](https://img.shields.io/pypi/v/polyrag.svg)](https://pypi.org/project/polyrag/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/polyrag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PolyRAG** is a modular, extensible, and production-ready Retrieval-Augmented Generation (RAG) framework designed to unify multiple RAG paradigms—from standard **Naive Vector RAG** and multi-hop **Agentic RAG**, to future expansions like **GraphRAG** and **Hybrid Retrieval**.

---

## 🌟 Key Features

- **Multi-Paradigm Pipelines**:
  - `NaiveRAG`: Standard retrieve-then-read pipeline with low latency.
  - `AgenticRAG`: Planning, semantic query rewriting, iterative multi-round vector search, deduplication, and fused reflection with grounded citations.
  - `ReActAgent`: Structured Thought-Action-Observation reasoning loop with dynamic tool execution.
  - *Coming Soon*: `GraphRAG` (knowledge graph entity extraction and community summarization).

- **Clean & Swappable Interfaces**:
  - **Chunkers**: `FixedSizeChunker`, `RecursiveCharacterChunker` (boundary-aware).
  - **Embeddings**: `OpenAIEmbedding`, `SentenceTransformerEmbedding`.
  - **Vector Stores**: `ChromaVectorStore` (persistent / in-memory), zero-dependency `InMemoryVectorStore`.
  - **LLMs**: `OpenAILLM` (supporting both Azure OpenAI response structures and standard chat completions).

- **Developer First**:
  - Zero-configuration `.from_env()` and `.create()` factory constructors.
  - 1-line upgrade from Naive to Agentic via `.as_agentic()`.
  - Fully typed with PEP 561 compliance (`py.typed`).

---

## 📚 Documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

- **[Getting Started](docs/getting-started/)**: [Installation](docs/getting-started/installation.md) & [Quickstart](docs/getting-started/quickstart.md)
- **[Configuration](docs/configuration/)**: [Environment Variables](docs/configuration/environment.md) & [Custom Pipelines](docs/configuration/custom-pipeline.md)
- **[Guides](docs/guides/)**: [Naive RAG](docs/guides/naive-rag.md), [Agentic RAG](docs/guides/agentic-rag.md), & [ReAct Agent](docs/guides/react-agent.md)
- **[API Reference](docs/api-reference/)**: [Core Entities](docs/api-reference/core.md) & [Services](docs/api-reference/service.md)
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

# Retrieve top-k relevant chunks
results = service.retrieve("What is DNS?", top_k=3)

# End-to-end question answering
response = service.query("How does DNS map names to addresses?")
print("Answer:", response.answer)
print("Sources:", response.sources)
```

---

### 2. Seamless Upgrade to Agentic RAG

Turn your existing retrieval service into an autonomous multi-round reasoning agent with one line:

```python
# Convert existing service into an Agentic RAG service
agentic = service.as_agentic(max_rounds=2, top_k=3, verbose=True)

# Multi-hop question answering with reflection & source citations
response = agentic.query(
    "What transport protocol does HTTP/3 rely on, and how does it handle connection migration?"
)

print("Grounded Answer:\n", response.answer)
print("Confidence:", response.confidence)
print("Execution Summary:", response.reasoning_summary)
```

---

### 3. Custom Pipeline Composition

Swap out any component without changing your core application logic:

```python
from polyrag import (
    NaiveRAG,
    RecursiveCharacterChunker,
    SentenceTransformerEmbedding,
    InMemoryVectorStore,
    OpenAILLM,
)

# Build a fully customized pipeline
pipeline = NaiveRAG(
    chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50),
    embedding_model=SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2"),
    vector_store=InMemoryVectorStore(),
    llm_client=OpenAILLM(model_name="gpt-4o-mini"),
)

pipeline.ingest_text("Custom document content...", source="my_doc.txt")
result = pipeline.execute("My question?")
print(result.answer)
```

---

## 🧪 Testing

PolyRAG includes a complete suite of unit tests verifying all chunkers, stores, and pipelines:

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
