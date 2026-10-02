# Advanced RAG Guide ⚡

The **Advanced RAG** (`polyrag.AdvancedRAG`) pipeline enhances `BaseRAG` with **Pre-Retrieval** and **Post-Retrieval** optimizations to overcome vector space sensitivity and lexical mismatch.

---

## 1. Architectural Workflow

```
[User Question]
       │
       ▼ (1. Pre-Retrieval: Query Expansion)
[Expanded Queries Q1, Q2, Q3]
       │
       ▼ (2. Multi-Query Parallel Retrieval)
[Search Run 1]    [Search Run 2]    [Search Run 3]
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
        (3. Post-Retrieval: Reciprocal Rank Fusion)
              RRF Score = ∑ 1 / (60 + rank)
                         │
                         ▼
             [Fused & Re-Ranked Chunks]
                         │
                         ▼ (4. Context Assembly & Prompting)
                 [Grounded LLM Answer]
```

---

## 2. Key Optimizations

### Pre-Retrieval: Multi-Query Expansion
Users rarely phrase questions using the exact terminology present in technical documents. `AdvancedRAG` generates $N$ alternative queries capturing:
- Technical synonyms and acronyms
- Rephrased sentence structures
- Sub-domain keywords

### Post-Retrieval: Reciprocal Rank Fusion (RRF)
Instead of relying on a single vector distance score, PolyRAG merges candidate lists using Reciprocal Rank Fusion:

$$RRF(d) = \sum_{q \in Q} \frac{1}{k + \text{rank}(d, q)}$$

- $k = 60$ (standard smoothing constant).
- Documents appearing in multiple query result sets rise to the top.
- Suppresses false-positive chunk matches that scored high on only one outlier query.

---

## 3. Code Example

```python
from polyrag import (
    AdvancedRAG,
    InMemoryVectorStore,
    PolyRAG,
    resolve_embedding_model,
    ChatOpenAI,
)

# Option A: Standalone pipeline
advanced = AdvancedRAG(
    embedding_model=resolve_embedding_model("openai"),
    vector_store=InMemoryVectorStore(),
    chat_model=ChatOpenAI(model="gpt-4o-mini"),
    num_expanded_queries=3,
    top_k=5,
    min_relevance_score=0.1,
    verbose=True,
)

# Option B: Creating from an existing PolyRAG application context in 1 line
app = PolyRAG.from_env()
advanced = app.create_advanced_rag(num_expanded_queries=3, top_k=5)

# Execute query
response = advanced.query("What mechanism does DNS use to handle UDP packet overflow?")

print("Answer:", response.answer)
print("Execution Summary:", response.reasoning_summary)
print("LLM Calls Made:", response.llm_calls)
```
