# Installation & Setup Guide

Welcome to **PolyRAG**! This guide covers installation options, optional dependency bundles, virtual environment configuration, and verification.

---

## 1. System Requirements

- **Python**: `>= 3.10` (tested on Python `3.10`, `3.11`, and `3.12`)
- **Operating System**: Linux, macOS, or Windows
- **Package Manager**: `pip` (version `21.0+` recommended)

---

## 2. Basic Installation

Install the lightweight core package with zero external database dependencies:

```bash
pip install polyrag
```

The core installation includes:
- The full `BaseRAG` class hierarchy (`NaiveRAG`, `AdvancedRAG`, `AgenticRAG`, `ReActAgent`).
- The Dependency Injection `Container`.
- Document chunkers (`FixedSizeChunker`, `RecursiveCharacterChunker`).
- Zero-dependency `InMemoryVectorStore` using cosine similarity.
- Data transfer models (`Document`, `Chunk`, `SearchResult`, `RAGResponse`, `AgentResponse`).

---

## 3. Optional Dependency Bundles

PolyRAG uses optional extras so your production builds stay slim. Install only what you need:

| Extra | Command | Description |
| :--- | :--- | :--- |
| **`[openai]`** | `pip install "polyrag[openai]"` | Official OpenAI SDK for LLM completions & embeddings |
| **`[chroma]`** | `pip install "polyrag[chroma]"` | ChromaDB vector database for persistent local or remote collections |
| **`[milvus]`** | `pip install "polyrag[milvus]"` | Milvus & Milvus Lite vector database (`pymilvus>=2.4.0`) |
| **`[embeddings]`** | `pip install "polyrag[embeddings]"` | HuggingFace `sentence-transformers` & `torch` for offline local embeddings |
| **`[all]`** | `pip install "polyrag[all]"` | Full production bundle with all vector stores, LLMs, and embeddings |
| **`[dev]`** | `pip install "polyrag[dev]"` | Developer tools (`pytest`, `build`, `twine`) |

---

## 4. Setting up a Virtual Environment

It is recommended to run PolyRAG in an isolated virtual environment:

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "polyrag[all]"
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install "polyrag[all]"
```

---

## 5. Development Mode (Editable Install)

To develop or contribute to PolyRAG:

```bash
git clone https://github.com/NVKQ2022/PolyRAG.git
cd PolyRAG

# Create environment and install in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"

# Run unit tests to verify setup
pytest tests
```

---

## 6. Verifying Installation

Verify that the library and CLI imports resolve correctly:

```bash
python -c "import polyrag; print(f'PolyRAG {polyrag.__version__} successfully imported!')"
```
