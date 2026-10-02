# Agentic RAG Guide 🧠

**Agentic RAG** (`polyrag.AgenticRAG`) elevates static retrieval into an autonomous cognitive reasoning loop. Rather than blindly executing a single search, the agent **plans**, **rewrites queries**, **retrieves across multiple rounds**, and performs **fused self-reflection**.

---

## 1. The 4 Core Pillars

```
[User Question]
       │
       ▼ (Call 1: Planning & Routing)
Is retrieval needed?
  ├── NO  ──► Direct Answer (Bypass retrieval for greetings/conversational queries)
  └── YES ──► Rewrite query with dense domain keywords
                     │
         ┌───────────┴────────────────────────┐
         ▼                                    │
    (Round N <= max_rounds)                   │
    [Vector Search & Deduplication]           │
         │                                    │
         ▼                                    │
    (Call 2: Fused Reflection & Synthesis)    │
    Are accumulated facts sufficient?         │
      ├── YES ──► Output Grounded Answer with [source#id] citations
      └── NO  ──► Extract missing gaps, refine query, loop back
```

### 1. Planning & Semantic Query Routing
The planner determines whether the question requires document retrieval. Polite greetings (e.g. *"Hello! How can you help me?"*) bypass vector search completely, returning an instant conversational answer. For technical queries, it rewrites the prompt into a keyword-dense semantic query.

### 2. Iterative Multi-Round Search & Deduplication
For complex multi-hop queries (where answers require combining information from multiple different chapters or RFCs), the agent retrieves evidence across up to `max_rounds`. Chunks are deduplicated across rounds by `(source, chunk_id)` to keep the prompt context window clean and unpolluted.

### 3. Fused Reflection & Grounded Synthesis
Standard Agentic RAG architectures invoke separate LLM calls to evaluate factual completeness and then synthesize the final text. PolyRAG unifies this into a **single fused prompt**:
- If evidence is complete (or the round limit is reached): The model outputs the final grounded answer with strict `[source#chunk_id]` citations immediately.
- If incomplete: The model outputs detected missing gaps and a refined `next_query`.
- **Result**: Cuts total LLM calls by **~35% to 50%**, reducing both latency and cost.

### 4. Observable Trajectory & Metrics
Every decision, latency measurement, and retrieved chunk is recorded in `agent_log` for transparent debugging.

---

## 2. Code Example

from polyrag import AgenticRAG, PolyRAG

# 1. Instantiate via PolyRAG application context
app = PolyRAG.from_env()

# Ingest multi-hop documents
app.ingest("HTTP/3 is built on the QUIC transport protocol (RFC 9000).", source="rfc9114.txt")
app.ingest("QUIC handles connection migration using Connection IDs across IP changes.", source="rfc9000.txt")

# 2. Create Agentic RAG
agentic = app.create_agentic_rag(max_rounds=2, top_k=3, verbose=True)

# 3. Multi-hop query requiring information from both documents
response = agentic.query(
    "What transport protocol does HTTP/3 rely on, and how does that protocol handle connection migration?"
)

# 4. Results
print("\n=== FINAL GROUNDED SYNTHESIS ===")
print(response.answer)

print("\n=== METRICS ===")
print(f"Confidence: {response.confidence:.2f}")
print(f"Total Latency: {response.took_ms} ms")
print(f"LLM Calls: {response.llm_calls}")

print("\n=== OBSERVABLE AGENT TRAJECTORY ===")
for action in response.agent_log:
    print(f">> Action: {action['action']:<25} | Took: {action.get('took_ms', 0)} ms")
```
