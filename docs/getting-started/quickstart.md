# Quickstart Guide 🚀

Get up and running with **PolyRAG** in under 5 minutes.

---

## 1. Minimal Working Example (Zero External API Keys)

PolyRAG includes an `InMemoryVectorStore` and supports offline embedding models. You can test retrieval logic without needing third-party API credentials:

```python
from polyrag import (
    NaiveRAG,
    RecursiveCharacterChunker,
    SentenceTransformerEmbedding,
    InMemoryVectorStore,
)

# 1. Initialize components
chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30)
embedding_model = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore()

# 2. Build the pipeline
rag = NaiveRAG(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
)

# 3. Ingest documents
sample_document = """
The Domain Name System (DNS) is a hierarchical naming system for computers,
services, or other resources connected to the Internet or a private network.
It translates human-friendly domain names like example.com into numerical IP addresses.
"""

rag.ingest_text(text=sample_document, source="dns_overview.txt")

# 4. Search relevant passages
results = rag.retrieve("How does DNS translate domain names?", top_k=2)

for r in results:
    doc = r["document"]
    print(f"[{doc['source']}#{doc['chunk_id']}] (Score: {r['score']:.4f})")
    print(doc["text"])
```

---

## 2. End-to-End RAG with `RAGService` (Using OpenAI / LLM)

When an LLM client is available, use `RAGService` for complete question-answering with factual grounding:

```python
import os
from polyrag import RAGService

# Set your API credentials
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["MODEL_NAME"] = "gpt-4o-mini"

# Initialize RAGService automatically from environment
service = RAGService.from_env()

# Ingest technical specifications or documentation
service.ingest_file("data/specifications.txt")

# Query the pipeline
response = service.query("What are the key requirements for authentication?")

print("Answer:", response.answer)
print("Context used:", response.context)
print("Sources:", response.sources)
```

---

## 3. Upgrading to Agentic Multi-Round RAG

If questions require multi-hop reasoning or missing evidence detection, upgrade your service instance with `.as_agentic()`:

```python
# Convert to Agentic RAG with reflection and multi-round retrieval
agentic_service = service.as_agentic(max_rounds=2, top_k=3, verbose=True)

# Complex multi-hop query
response = agentic_service.query(
    "How does TLS 1.3 key exchange differ from TLS 1.2, and how does QUIC incorporate it?"
)

print("\n=== FINAL GROUNDED SYNTHESIS ===")
print(response.answer)
print("\n=== CONFIDENCE ===", response.confidence)
print("=== REASONING ===", response.reasoning_summary)
```

---

## Next Steps

- Explore [Configuration Guide](../configuration/environment.md) to customize vector databases, chunking sizes, and models.
- Learn about [Agentic RAG Workflows](../guides/agentic-rag.md) for self-reflection and multi-hop queries.
- Check out the [ReAct Agent Guide](../guides/react-agent.md) for tool-based reasoning.
