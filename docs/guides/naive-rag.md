# Naive RAG Guide 📖

The **Naive RAG** (Retrieve-then-Read) pipeline is the foundation of Retrieval-Augmented Generation. It provides a straightforward, low-latency workflow suitable for single-hop factual lookups.

---

## Architecture Flow

```
[Input Document]
       │
       ▼ (1. Chunk)
[Document Chunks]
       │
       ▼ (2. Embed)
[Embedding Vectors] ──► Stored in Vector Database
                                 │
[User Question]                  │
       │                         │
       ▼                         │
[Query Vector] ────────► (3. Similarity Search)
                                 │
                                 ▼
                     [Top-k Retrieved Chunks]
                                 │
                                 ▼ (4. Format Context)
                     [Prompt Context Assembly]
                                 │
                                 ▼ (5. Generation)
                         [Grounded Answer]
```

---

## Usage Example

```python
from polyrag import (
    NaiveRAG,
    RecursiveCharacterChunker,
    SentenceTransformerEmbedding,
    InMemoryVectorStore,
    OpenAILLM,
)

# 1. Initialize Pipeline
rag = NaiveRAG(
    chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50),
    embedding_model=SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2"),
    vector_store=InMemoryVectorStore(),
    llm_client=OpenAILLM(model_name="gpt-4o-mini"),
)

# 2. Ingest Document
rag.ingest_text(
    text="""RFC 791 describes the Internet Protocol (IP). 
The minimum IPv4 header size is 20 bytes without options, 
and the maximum header size is 60 bytes when options are present.""",
    source="rfc791.txt",
    metadata={"standard": "IETF", "protocol": "IPv4"},
)

# 3. Retrieve Context
chunks = rag.retrieve("What is the IPv4 header size?", top_k=2)

# 4. Generate Answer
response = rag.execute("What is the minimum IPv4 header size?", top_k=2)

print("Answer:", response.answer)
print("Confidence:", response.confidence)
print("Context:", response.context)
```

---

## When to Use Naive RAG

✅ **Ideal For**:
- Single-hop factual questions (e.g., *"What is the port number for HTTPS?"*).
- Simple document lookups with clear keyword overlap.
- Low-latency requirements (only 1 retrieval + 1 LLM completion).

❌ **When to Upgrade to Agentic RAG**:
- Multi-hop questions requiring evidence from multiple distinct sections or documents.
- Ambiguous user queries requiring keyword expansion.
- Cases where verification of evidence sufficiency is needed to eliminate hallucinations.
