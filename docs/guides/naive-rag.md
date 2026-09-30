# Naive RAG Guide 📖

The **Naive RAG** (`polyrag.NaiveRAG`) pipeline is the direct implementation of the foundational **Retrieve-then-Read** paradigm. It inherits core ingestion and vector search from `BaseRAG` and executes a single retrieval step followed by LLM prompt generation.

---

## 1. Pipeline Flow

```
[Input Document]
       │
       ▼ (1. Chunk via BaseRAG)
[Document Chunks]
       │
       ▼ (2. Embed via BaseRAG)
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

## 2. Code Example

```python
from polyrag import (
    NaiveRAG,
    RecursiveCharacterChunker,
    SentenceTransformerEmbedding,
    InMemoryVectorStore,
    OpenAILLM,
)

# 1. Instantiate Pipeline
rag = NaiveRAG(
    chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50),
    embedding_model=SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2"),
    vector_store=InMemoryVectorStore(),
    llm_client=OpenAILLM(model_name="gpt-4o-mini"),
)

# 2. Ingest Technical Content
rag.ingest_text(
    text="""RFC 791 defines the Internet Protocol. The minimum IPv4 header
is 20 bytes and can reach a maximum of 60 bytes when IP options are used.""",
    source="rfc791.txt",
    metadata={"layer": "network"},
)

# 3. Execute Query
response = rag.query("What is the maximum size of an IPv4 header?", top_k=2)

print("Answer:", response.answer)
print("Confidence:", response.confidence)
print("Sources Cited:", response.sources)
```

---

## 3. Characteristics

- **Latency**: Minimal (~1 vector search + 1 LLM completion).
- **Cost**: 1 LLM call per query.
- **Best Use Case**: Direct, single-hop factual questions with high semantic overlap between question and document passages.
- **When to upgrade**: If questions require combining facts from disparate pages or documents, upgrade to [`AdvancedRAG`](advanced-rag.md) or [`AgenticRAG`](agentic-rag.md).
