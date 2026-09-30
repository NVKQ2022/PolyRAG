# Installation Guide

This guide walks you through installing **PolyRAG** and configuring its optional dependencies for various environments.

---

## Prerequisites

- **Python**: Version `3.10`, `3.11`, or `3.12`.
- **Package Manager**: `pip` (version 21.0 or higher recommended).

---

## Basic Installation

To install the minimal PolyRAG package (includes core interfaces, dataclasses, and zero-dependency `InMemoryVectorStore`):

```bash
pip install polyrag
```

---

## Optional Dependency Bundles

PolyRAG is designed to be lightweight by default. Depending on the vector stores, embedding providers, or LLM clients you use, install the relevant extra bundle:

### 1. OpenAI LLM & Embeddings Support
Includes official `openai` SDK bindings:
```bash
pip install "polyrag[openai]"
```

### 2. ChromaDB Vector Database
Includes `chromadb` client for persistent disk storage or client/server mode:
```bash
pip install "polyrag[chroma]"
```

### 3. Local SentenceTransformers & Torch
Includes `sentence-transformers` and `torch` for local, offline embeddings without API costs:
```bash
pip install "polyrag[embeddings]"
```

### 4. Full Production Bundle
Installs all supported adapters, vector stores, embedding models, and API frameworks:
```bash
pip install "polyrag[all]"
```

### 5. Development & Testing Tools
Includes `pytest`, `build`, and `twine` for contributors:
```bash
pip install "polyrag[dev]"
```

---

## Installing from Source (Editable Mode)

For local development or contributing to PolyRAG:

```bash
# 1. Clone the repository
git clone https://github.com/NVKQ2022/PolyRAG.git
cd PolyRAG

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install in editable mode with development dependencies
pip install -e ".[all,dev]"

# 4. Verify test suite passes
pytest tests
```

---

## Verification

Run a quick Python command to verify that PolyRAG is correctly installed:

```bash
python -c "import polyrag; print(f'PolyRAG {polyrag.__version__} successfully installed!')"
```
