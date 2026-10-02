# Quickstart Guide 🚀

Get started with **PolyRAG** in under 5 minutes. This guide walks you through zero-dependency prototyping, production service facades, and the RAG class hierarchy.

---

## 1. Zero-Dependency Prototyping (No API Key Required)

PolyRAG includes an `InMemoryVectorStore` and supports offline local embeddings. You can index, retrieve, and test retrieval logic right away:

```python
from polyrag import (
    NaiveRAG,
    RecursiveCharacterChunker,
    InMemoryVectorStore,
    resolve_embedding_model,
)

# 1. Initialize components (zero API key / zero dependency local mode)
chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=40)
embedding_model = resolve_embedding_model(None)  # Zero-setup local FakeEmbeddings
vector_store = InMemoryVectorStore()

# 2. Assemble NaiveRAG pipeline
rag = NaiveRAG(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
)

# 3. Ingest documents
rag.ingest_text(
    text="""The Domain Name System (DNS) translates human-readable domain names 
(like example.com) to machine-readable IP addresses (like 93.184.216.34).
DNS uses UDP port 53 by default, but falls back to TCP when responses exceed 512 bytes.""",
    source="rfc1035_summary.txt",
)

# 4. Perform vector search
results = rag.retrieve("When does DNS use TCP port 53?", top_k=2)

for r in results:
    doc = r["document"]
    print(f"[{doc['source']}#{doc['chunk_id']}] Similarity: {r['score']:.4f}")
    print(doc["text"])
```

---

## 2. Central Application Context (`PolyRAG`)

`PolyRAG` serves as the central setup orchestrator and pipeline factory:
- Configures default embeddings, vector store, and LLM from `.env` in 1 line.
- Manages shared document ingestion into the underlying vector store.
- Manufactures specialized pipelines ([`NaiveRAG`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/pipelines/naive.py), [`AdvancedRAG`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/pipelines/advanced.py), [`AgenticRAG`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/pipelines/agentic.py), [`ReActAgent`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/pipelines/react.py)).

```python
import os
from polyrag import PolyRAG

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["MODEL_NAME"] = "gpt-4o-mini"

# Automatic initialization from environment
app = PolyRAG.from_env()

# Ingest single files or entire folders into the shared store
app.ingest_file("data/architecture_spec.txt")
app.ingest_directory("docs/", glob_pattern="*.txt")

# End-to-end question answering (convenience shortcut)
response = app.query("What are the main communication protocols?")
print("Answer:", response.answer)
print("Sources:", response.sources)
```

---

## 3. Scaling Through the RAG Hierarchy

As question complexity grows, manufacture specialized pipelines from the configured `PolyRAG` context:

```python
# Level 1: Standard 1-Shot RAG
naive = app.create_naive_rag()
naive_response = naive.query("What is DNS?")

# Level 2: Advanced RAG (Multi-Query Expansion & RRF Re-ranking)
advanced = app.create_advanced_rag(num_expanded_queries=3, top_k=5)
advanced_response = advanced.query("How does DNS handle packet truncation?")

# Level 3: Agentic RAG (Autonomous Planning, Multi-Round Loop & Fused Reflection)
agentic = app.create_agentic_rag(max_rounds=2, top_k=3, verbose=True)
agentic_response = agentic.query(
    "Compare how DNS and QUIC handle packet fallback and connection recovery."
)

print("Agentic Grounded Answer:\n", agentic_response.answer)
print("Confidence:", agentic_response.confidence)
print("Trajectory:", agentic_response.reasoning_summary)
```

---

## 4. Using the Dependency Injection Container

For clean architecture and enterprise projects, use `Container`:

```python
from polyrag import Container

# Build from environment
container = Container.from_env()

# Resolve pre-wired pipelines
pipeline = container.build_agentic_rag(top_k=3, max_rounds=2)
response = pipeline.query("Explain TLS 1.3 0-RTT handshakes.")
print(response.answer)
```
