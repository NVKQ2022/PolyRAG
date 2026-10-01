# Module: `polyrag.pipelines` 🚀

The `polyrag.pipelines` module houses the RAG workflow engines. All pipelines inherit from the abstract base class `BaseRAG`, ensuring identical ingestion and retrieval interfaces across all paradigms.

---

## 1. Class Hierarchy

```
                     ┌─────────────────────────┐
                     │         BaseRAG         │
                     │  (Abstract Base Class)  │
                     └────────────┬────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│   NaiveRAG   │           │ AdvancedRAG  │           │  AgenticRAG  │
│ (1-Shot RAG) │           │ (Expand/RRF) │           │ (Reflection) │
└──────────────┘           └──────────────┘           └──────┬───────┘
                                                             │
                                                             ▼
                                                      ┌──────────────┐
                                                      │  ReActAgent  │
                                                      │ (ReAct Loop) │
                                                      └──────────────┘
```

---

## 2. Common Inherited Primitives (`BaseRAG`)

All pipelines share the following ingestion and retrieval methods:

```python
# Ingest raw text directly
rag.ingest_text(
    text="Authentication token expires after 3600 seconds.",
    source="auth_policy.md",
    metadata={"department": "Security"},
)

# Ingest single file from disk
rag.ingest_file(file_path="docs/architecture.pdf")

# Recursively ingest an entire folder
rag.ingest_directory(dir_path="knowledge_base/", glob_pattern="**/*.md")

# Low-level nearest-neighbor retrieval
results = rag.retrieve(query="What is the token TTL?", top_k=3)

# Unified question answering
response = rag.query("How long does the auth token last?")
print(response.answer)
print(response.sources)
```

---

## 3. Pipeline Implementations

### A. `NaiveRAG` (1-Shot Retrieve-then-Read)
The standard vector RAG baseline: embeds the query, searches nearest neighbors, builds a context prompt, and generates an answer in a single LLM invocation.

```python
from polyrag.pipelines import NaiveRAG

naive_rag = NaiveRAG(
    embedding_model=embedder,
    vector_store=vector_store,
    llm_client=llm,
    chunker=chunker,
)
res = naive_rag.query("What is the primary function of DNS?")
```

---

### B. `AdvancedRAG` (Expansion & RRF)
Improves retrieval quality through pre- and post-retrieval optimization:
1. **Multi-Query Expansion**: LLM rewrites the user query into multiple alternative perspectives.
2. **Parallel Retrieval**: Runs vector similarity search across all rewritten queries.
3. **Reciprocal Rank Fusion (RRF)**: Re-ranks candidates using the formula $\sum \frac{1}{60 + \text{rank}}$.
4. **Relevance Thresholding**: Filters out low-confidence chunks (`min_relevance_score`).

```python
from polyrag.pipelines import AdvancedRAG

advanced_rag = AdvancedRAG(
    embedding_model=embedder,
    vector_store=vector_store,
    llm_client=llm,
    top_k=5,
    num_expanded_queries=3,
    min_relevance_score=0.4,
    verbose=True,
)
res = advanced_rag.query("How does DNS handle zone transfers?")
```

---

### C. `AgenticRAG` (Autonomous Self-Reflection)
Designed for complex questions requiring multi-round investigation:
1. **Intent Classification & Conversational Bypass**: Answers greetings and chitchat directly without wasting vector queries.
2. **Semantic Keyword Rewriting**: Generates optimized search queries instead of raw natural language.
3. **Iterative Multi-Round Retrieval**: Performs up to `max_rounds` of search, deduplicating findings across rounds.
4. **Fused Self-Reflection**: Verifies whether the retrieved context contains sufficient evidence to answer truthfully before generating the final response.

```python
from polyrag.pipelines import AgenticRAG

agentic_rag = AgenticRAG(
    embedding_model=embedder,
    vector_store=vector_store,
    llm_client=llm,
    top_k=3,
    max_rounds=2,
    verbose=True,
)
res = agentic_rag.query("Compare HTTP/2 multiplexing with HTTP/3 QUIC connection migration.")
print(res.confidence)
print(res.reasoning_summary)
```

---

### D. `ReActAgent` (Tool-Driven Reasoning Loop)
An autonomous agent using the **Thought $\rightarrow$ Action $\rightarrow$ Observation** loop:
* Includes built-in vector search tools (`search_knowledge_base`).
* Accepts arbitrary custom Python functions as tools.

```python
from polyrag.pipelines import ReActAgent
from polyrag.core.models import ToolDefinition

def check_server_status(host: str) -> str:
    return f"Host {host} is ONLINE (latency: 12ms)"

status_tool = ToolDefinition(
    name="check_server_status",
    description="Check whether a specific host or microservice is online.",
    parameters={"host": "Host name or IP"},
    func=check_server_status,
)

agent = ReActAgent(
    embedding_model=embedder,
    vector_store=vector_store,
    llm_client=llm,
    tools=[status_tool],
    max_steps=5,
    verbose=True,
)

res = agent.run("Why is api.domain.com failing, and is the host currently online?")
for step in res.steps:
    print(f"Thought: {step.thought}")
    print(f"Action:  {step.action}({step.action_input}) -> {step.observation}")
print(f"Final Answer: {res.answer}")
```
