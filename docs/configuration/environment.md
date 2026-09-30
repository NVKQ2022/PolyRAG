# Environment Configuration

PolyRAG supports declarative configuration through system environment variables or `.env` files. This allows zero-code configuration when using `RAGService.from_env()`.

---

## Supported Environment Variables

| Variable | Description | Default | Example |
| :--- | :--- | :--- | :--- |
| `OPENAI_API_KEY` | Secret API key for OpenAI or Azure OpenAI | *None* | `sk-proj-...` |
| `MODEL_NAME` | Target LLM model name | `gpt-4o-mini` | `gpt-4o-mini`, `gpt-4o` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint URL | *None* | `https://my-resource.openai.azure.com` |
| `OPENAI_API_VERSION` | API version for Azure OpenAI completions | *None* | `2024-02-15-preview` |
| `EMBEDDING_MODEL` | HuggingFace or OpenAI embedding model name | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2`, `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | Path to persistent local vector database folder | `./chroma_db` | `./data/chroma_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `documents` | `rfc_collection` |

---

## Example `.env` File

Create a `.env` file in the root of your application:

```bash
# LLM Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
MODEL_NAME=gpt-4o-mini

# Vector Database Configuration
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=technical_specs

# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## Using Environment Variables in Code

When calling `RAGService.from_env()`, PolyRAG automatically resolves your settings:

```python
from dotenv import load_dotenv
from polyrag import RAGService

# Load environment variables from .env
load_dotenv()

# Automatically creates clients matching the environment configuration:
service = RAGService.from_env(
    persist_dir="./chroma_db",
    collection_name="technical_specs",
    embedding_model="all-MiniLM-L6-v2",
)
```

---

## Azure OpenAI Configuration

If you are using Microsoft Azure OpenAI:

```bash
AZURE_OPENAI_ENDPOINT=https://your-company.openai.azure.com/
OPENAI_API_KEY=your-azure-key
OPENAI_API_VERSION=2024-02-15-preview
MODEL_NAME=gpt-4o-mini  # Your Azure deployment name
```

PolyRAG's `OpenAILLM` adapter detects both standard OpenAI and Azure OpenAI response contracts automatically.
