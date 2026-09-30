# Agentic RAG Guide 🧠

**Agentic RAG** elevates traditional retrieval by adding an autonomous cognitive feedback loop: planning, query rewriting, multi-round vector retrieval, deduplication, and fused self-reflection.

---

## The 4 Core Pillars of PolyRAG's Agentic Engine

```
[User Question]
       │
       ▼ (Call 1: Planning & Routing)
Is retrieval needed?
  ├── NO  ──► Direct Answer (Bypass retrieval for greetings/chit-chat)
  └── YES ──► Rewrite query with dense technical keywords
                     │
         ┌───────────┴────────────────────────┐
         ▼                                    │
    (Round N <= max_rounds)                   │
    [Vector Search & Deduplication]           │
         │                                    │
         ▼                                    │
    (Call 2: Fused Reflection & Synthesis)    │
    Are accumulated facts sufficient?         │
      ├── YES ──► Output Grounded Answer [source#id]
      └── NO  ──► Identify missing gaps, refine query, loop back
```

### 1. Retrieval Planning & Query Rewriting
Before executing vector searches, the planner evaluates whether retrieval is needed. If true, it rewrites the user query to expand domain terminology and boost vector embedding similarity.

### 2. Iterative Multi-Round Search & Deduplication
For multi-hop queries where information is scattered across different documents, the agent retrieves evidence iteratively. It deduplicates chunks by `(source, chunk_id)` so the LLM context remains concise and unpolluted.

### 3. Fused Reflection & Answer Generation
Instead of making separate LLM calls to evaluate sufficiency and then synthesize answers, PolyRAG uses **Fused Reflection**:
- If evidence is sufficient (or final round is reached): generates the grounded answer citing `[source#chunk_id]` directly in the same LLM call.
- If evidence is insufficient: extracts missing gaps and outputs a targeted `next_query`.

### 4. Direct Conversational Bypass
Conversational queries or polite greetings bypass the vector database completely, saving latency and embedding computations.

---

## Code Example

```python
from polyrag import AgenticRAG, SentenceTransformerEmbedding, InMemoryVectorStore, OpenAILLM

# 1. Initialize components
emb = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
vdb = InMemoryVectorStore()
llm = OpenAILLM(model_name="gpt-4o-mini")

# 2. Instantiate AgenticRAG
agentic = AgenticRAG(
    embedding_model=emb,
    vector_store=vdb,
    llm_client=llm,
    top_k=3,
    max_rounds=2,
    verbose=True,  # Enables step-by-step observable logging
)

# 3. Multi-Hop Query Execution
response = agentic.query(
    "What underlying protocol does HTTP/3 rely on, and how does connection migration work?"
)

# 4. Inspect Observable Trajectory
print("\n=== FINAL ANSWER ===")
print(response.answer)

print("\n=== METRICS ===")
print("Confidence:", response.confidence)
print("Execution Time:", response.took_ms, "ms")
print("LLM Calls Made:", response.llm_calls)

print("\n=== AGENT OBSERVABLE LOG ===")
for action in response.agent_log:
    print(f"Action: {action['action']} | Took: {action.get('took_ms', 0)}ms")
```
