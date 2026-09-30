# Environment Configuration Reference

PolyRAG supports declarative configuration through system environment variables and `.env` files. Both `RAGService.from_env()` and `Container.from_env()` read these parameters automatically.

---

## 1. Supported Variables

| Environment Variable | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `OPENAI_API_KEY` | Secret API key for OpenAI / Azure OpenAI | *None* | `sk-proj-xyz...` |
| `MODEL_NAME` | Primary LLM model deployment name | `gpt-4o-mini` | `gpt-4o-mini`, `gpt-4o` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint URL | *None* | `https://my-res.openai.azure.com` |
| `OPENAI_API_VERSION` | API version for Azure OpenAI | *None* | `2024-02-15-preview` |
| `EMBEDDING_MODEL` | Embedding model identifier | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2`, `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | Local disk folder for ChromaDB storage | `./chroma_db` | `./data/chroma_db` |
| `COLLECTION_NAME` | Target vector collection name | `documents` | `rfc_documents` |

---

## 2. Sample `.env` Configuration

Create a `.env` file in the root of your project:

```bash
# ==========================================
# LLM Provider Configuration
# ==========================================
OPENAI_API_KEY=sk-your-openai-api-key
MODEL_NAME=gpt-4o-mini

# ==========================================
# Vector Database Configuration
# ==========================================
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=technical_specs

# ==========================================
# Embedding Model Configuration
# ==========================================
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 3. Microsoft Azure OpenAI Setup

For enterprise deployments on Microsoft Azure, set the endpoint and API version:

```bash
AZURE_OPENAI_ENDPOINT=https://your-company.openai.azure.com/
OPENAI_API_KEY=your-azure-api-key
OPENAI_API_VERSION=2024-02-15-preview
MODEL_NAME=gpt-4o-mini  # Your Azure deployment name
```

PolyRAG's `OpenAILLM` adapter detects whether `responses.create` or `chat.completions` is available and normalizes the output seamlessly.

---

## 4. Programmatic Usage

```python
from dotenv import load_dotenv
from polyrag import Container, RAGService

load_dotenv()

# Option A: Automatic via Facade
service = RAGService.from_env()

# Option B: Automatic via Dependency Injection Container
container = Container.from_env()
pipeline = container.build_agentic_rag()
```
