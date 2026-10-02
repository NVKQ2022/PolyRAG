# Module: `polyrag.vector_stores` 🗄️

The `polyrag.vector_stores` module manages nearest-neighbor vector indexing, document persistence, and semantic similarity search.

PolyRAG is built natively on LangChain's standard [`VectorStore`](https://python.langchain.com/docs/concepts/vectorstores/) interface. This means **any vector database in the LangChain ecosystem** can be plugged directly into PolyRAG pipelines with zero custom wrapper code.

---

## 1. Architecture: Universal LangChain VectorStore Support

PolyRAG pipelines accept any class that implements the standard LangChain `VectorStore` contract:

```
                  ┌──────────────────────────────┐
                  │    langchain.VectorStore     │
                  │   (Universal Standard Port)  │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│InMemoryVectorStore│  │langchain_chroma.  │   │langchain_milvus.  │
│(Zero-dependency) │   │     Chroma        │   │     Milvus        │
└──────────────────┘   └───────────────────┘   └───────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌──────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│      FAISS       │   │     Pinecone      │   │    PGVector /     │
│ (Local Flat/HNSW)│   │   (Cloud Server)  │   │     Qdrant        │
└──────────────────┘   └───────────────────┘   └───────────────────┘
```

---

## 2. Built-in: `InMemoryVectorStore`

Included directly in `polyrag` with **zero external dependencies**. Perfect for fast unit testing, ephemeral scripts, offline notebooks, and zero-setup prototyping.

```python
from polyrag.vector_stores import InMemoryVectorStore

# In-memory vector store with cosine distance
vdb = InMemoryVectorStore()

# Standard BaseVectorStore operations
vdb.add_documents(
    vectors=[[0.1, 0.2, 0.3], [0.8, 0.9, 0.7]],
    documents=[{"text": "Chunk 1", "source": "a.txt"}, {"text": "Chunk 2", "source": "b.txt"}],
)

results = vdb.search(query_vector=[0.1, 0.2, 0.3], top_k=1)
print(f"Top match: {results[0]['document']['text']}")
print(f"Total count: {vdb.count()}")
```

---

## 3. Production Vector Stores via Official LangChain Packages

### A. Chroma (`langchain-chroma`)
Local persistent SQLite and HNSW vector storage:

```bash
pip install langchain-chroma
```

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from polyrag import PolyRAG

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

rag = PolyRAG(vector_store=vector_store, embedding_model=embeddings)
```

---

### B. Milvus & Milvus Lite (`langchain-milvus`)
Scale from local file (`.db`) to planetary-scale distributed clusters:

```bash
pip install langchain-milvus
```

```python
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from polyrag import PolyRAG

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Mode 1: Local embedded Milvus Lite (no Docker required)
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "./data/milvus_demo.db"},
    collection_name="rfcs",
)

# Mode 2: Distributed Milvus cluster or Zilliz Cloud
# vector_store = Milvus(
#     embedding_function=embeddings,
#     connection_args={
#         "uri": "https://in03-xxxxxxxx.api.gcp-us-west1.zillizcloud.com",
#         "token": "YOUR_ZILLIZ_TOKEN",
#     },
#     collection_name="production_rfcs",
# )

rag = PolyRAG(vector_store=vector_store, embedding_model=embeddings)
```

---

### C. FAISS, Pinecone, Qdrant, PGVector
Simply install the respective LangChain package and pass the instance:

```python
# FAISS Example
from langchain_community.vectorstores import FAISS

faiss_store = FAISS.from_texts(
    texts=["Sample text"],
    embedding=embeddings,
)

rag = PolyRAG(vector_store=faiss_store, embedding_model=embeddings)
```

---

## 4. Automatic Resolution via `resolve_vector_store`

The `polyrag.vector_stores.resolve_vector_store` function handles string aliases and instances transparently:

```python
from polyrag.vector_stores import resolve_vector_store

# 1. Resolves directly if already a VectorStore instance
vdb = resolve_vector_store(existing_vdb)

# 2. Resolves 'chroma' by dynamically loading langchain_chroma.Chroma
vdb = resolve_vector_store("chroma", persist_directory="./chroma_db", embedding=embeddings)

# 3. Resolves 'milvus' by dynamically loading langchain_milvus.Milvus
vdb = resolve_vector_store("milvus", connection_args={"uri": "./milvus.db"}, embedding=embeddings)

# 4. Defaults to InMemoryVectorStore when None
vdb = resolve_vector_store(None)
```

---

## 5. Unified Interface Contract

All vector stores in PolyRAG conform to both LangChain standard methods and PolyRAG port methods:

```python
# PolyRAG Port Methods
vdb.add_documents(vectors=vectors, documents=documents, batch_size=5000)
results = vdb.search(query_vector=query_vector, top_k=5)
count = vdb.count()
sample = vdb.peek(limit=5)
vdb.clear()

# LangChain Standard Methods
lc_docs = vdb.similarity_search("query text", k=5)
docs_and_scores = vdb.similarity_search_with_score("query text", k=5)
```
