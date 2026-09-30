# Advanced RAG Guide ⚡

**Advanced RAG** extends the foundational `BaseRAG` pipeline by introducing **Pre-Retrieval** (Multi-Query Expansion) and **Post-Retrieval** (Reciprocal Rank Fusion and Re-ranking) strategies to overcome vector embedding blind spots and the "Lost in the Middle" phenomenon.

---

## The RAG Class Hierarchy

PolyRAG implements a strict object-oriented class hierarchy rooted in `BaseRAG`:

```
                    ┌─────────────────────────┐
                    │         BaseRAG         │
                    │  (Abstract Base Class)  │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
  │   NaiveRAG   │        │ AdvancedRAG  │        │  AgenticRAG  │
  │ (1-Shot RAG) │        │ (Expand/RRF) │        │ (Reflection) │
  └──────────────┘        └──────────────┘        └──────┬───────┘
                                                         │
                                                         ▼
                                                  ┌──────────────┐
                                                  │  ReActAgent  │
                                                  │(Tool Action) │
                                                  └──────────────┘
```

All derived classes inherit common document ingestion (`ingest_text`, `ingest_file`, `ingest_directory`), vector search (`retrieve`), and context formatting (`format_context`) from `BaseRAG`.

---

## 🌟 Advanced RAG Optimizations

### 1. Pre-Retrieval: Multi-Query Expansion
Natural language queries can miss relevant chunks due to wording differences. `AdvancedRAG` generates multiple diverse search queries capturing synonyms and technical perspectives before searching:

```
"How to secure HTTP?" ──► [ "How to secure HTTP?",
                            "TLS HTTPS encryption handshake",
                            "RFC 8446 transport security best practices" ]
```

### 2. Retrieval: Multi-Query Search
Executes vector searches across all generated query variations.

### 3. Post-Retrieval: Reciprocal Rank Fusion (RRF)
Merges ranked candidate lists from all queries using RRF:

$$\text{RRF Score}(d) = \sum_{q} \frac{1}{k + \text{rank}(d, q)}$$

Where $k = 60$. Items appearing across multiple query runs receive higher confidence and rank top in the prompt context.

---

## Usage Example

```python
from polyrag import (
    AdvancedRAG,
    SentenceTransformerEmbedding,
    InMemoryVectorStore,
    OpenAILLM,
    RAGService,
)

# 1. Direct instantiation
advanced = AdvancedRAG(
    embedding_model=SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2"),
    vector_store=InMemoryVectorStore(),
    llm_client=OpenAILLM(model_name="gpt-4o-mini"),
    num_expanded_queries=3,
    top_k=5,
)

# 2. Or upgrade an existing RAGService instance in 1 line:
service = RAGService.from_env()
advanced = service.as_advanced(num_expanded_queries=3, top_k=5)

# 3. Query
response = advanced.query("Explain connection migration in QUIC.")
print(response.answer)
print(response.reasoning_summary)
```
