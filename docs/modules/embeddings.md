# Module: `polyrag.embeddings` 🧬

The `polyrag.embeddings` module provides dense vector encoding for text chunks and search queries.

PolyRAG is built natively on LangChain's [`Embeddings`](https://python.langchain.com/docs/concepts/embedding_models/) interface. You can plug in **any embedding model from the LangChain ecosystem** (OpenAI, HuggingFace, Ollama, Cohere, Bedrock, VertexAI) with zero wrapper code.

---

## 1. Universal LangChain Embeddings Integration

Any class conforming to LangChain's `Embeddings` protocol (`embed_documents` and `embed_query`) works directly in PolyRAG:

```python
# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
openai_emb = OpenAIEmbeddings(model="text-embedding-3-small")

# Local HuggingFace Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
hf_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Local Ollama Embeddings
# from langchain_community.embeddings import OllamaEmbeddings
# ollama_emb = OllamaEmbeddings(model="nomic-embed-text")
```

---

## 2. Zero-Setup Local Development: `FakeEmbeddings`

When you want to prototype or run unit tests without downloading weights or making API calls, PolyRAG provides `FakeEmbeddings` with deterministic dimension:

```python
from polyrag.embeddings import resolve_embedding_model

# Generates deterministic synthetic embeddings of dimension 384
emb = resolve_embedding_model("fake", size=384)

vec = emb.embed_query("Sample text")
print(len(vec))  # 384
```

---

## 3. Automatic Resolution via `resolve_embedding_model`

PolyRAG provides a flexible resolver function:

```python
from polyrag.embeddings import resolve_embedding_model

# 1. Defaults to zero-setup FakeEmbeddings(size=384) if None
emb = resolve_embedding_model(None)

# 2. String alias for OpenAI
# Automatically instantiates langchain_openai.OpenAIEmbeddings
emb = resolve_embedding_model("openai")
emb = resolve_embedding_model("text-embedding-3-small")

# 3. String alias for HuggingFace / Local
# Automatically instantiates langchain_huggingface.HuggingFaceEmbeddings
emb = resolve_embedding_model("all-MiniLM-L6-v2")

# 4. Direct LangChain Embeddings instance
emb = resolve_embedding_model(my_langchain_embeddings)

# 5. Custom Duck-typed class
class MyCustomEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 128
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 128 for _ in texts]

emb = resolve_embedding_model(MyCustomEmbedder())
```

---

## 4. `BaseEmbeddingModel` Interface Contract

PolyRAG defines `BaseEmbeddingModel`, which subclasses LangChain's `Embeddings` while supporting both LangChain and PolyRAG ports:

| Method / Property | Source | Purpose |
| :--- | :--- | :--- |
| `dim -> int` | PolyRAG | Returns vector dimensionality (e.g. 384, 1536). |
| `embed_text(text: str) -> list[float]` | PolyRAG Port | Single text embedding generation. |
| `embed_batch(texts: list[str]) -> list[list[float]]` | PolyRAG Port | Batch embedding generation. |
| `embed_query(text: str) -> list[float]` | LangChain Standard | Query text embedding generation. |
| `embed_documents(texts: list[str]) -> list[list[float]]` | LangChain Standard | Document batch embedding generation. |
