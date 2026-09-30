# ReAct Agent Guide 🤖

The **ReAct Agent** (`polyrag.ReActAgent` / `polyrag.ReActRAG`) implements the **Reasoning + Acting** state machine (Yao et al., 2022). It inherits from `BaseRAG` and empowers the LLM to dynamically determine its own retrieval trajectory by invoking tools.

---

## 1. The Thought ➔ Action ➔ Observation Cycle

```
[User Question]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                   ReAct Execution Loop                      │
│                                                             │
│  Step 1:                                                    │
│    • Thought: "I need to find the TCP handshake mechanism." │
│    • Action:  search({"query": "TCP 3-way handshake SYN"})  │
│    • Observation: [rfc793.txt#4] retrieved 2 chunks.       │
│                                                             │
│  Step 2:                                                    │
│    • Thought: "I have gathered enough facts to answer."     │
│    • Action:  final_answer({"answer": "...", "conf": 0.95}) │
│                                                             │
│  (Repeats up to max_steps or until final_answer emitted)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                   [AgentResponse Object]
```

---

## 2. Built-In Tools

| Tool Name | Action Input Schema | Description |
| :--- | :--- | :--- |
| **`search`** | `{"query": str, "top_k": int}` | Executes nearest-neighbor semantic search in the vector knowledge base. |
| **`list_docs`** | `{}` | Returns collection statistics and preview sample records. |
| **`final_answer`** | `{"answer": str, "confidence": float}` | Concludes the reasoning loop and emits the grounded answer with citations. |

---

## 3. Code Example

```python
from polyrag import (
    ReActAgent,
    SentenceTransformerEmbedding,
    InMemoryVectorStore,
    OpenAILLM,
)

# 1. Initialize dependencies
emb = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
vdb = InMemoryVectorStore()
llm = OpenAILLM(model_name="gpt-4o-mini")

# 2. Ingest document via BaseRAG interface
agent = ReActAgent(
    llm_client=llm,
    embedding_model=emb,
    vector_store=vdb,
    max_steps=4,
    default_top_k=3,
    verbose=True,
)

agent.ingest_text(
    text="RFC 793 defines the TCP three-way handshake: SYN, SYN-ACK, ACK.",
    source="rfc793.txt",
)

# 3. Execute reasoning loop
response = agent.query("Explain the TCP three-way handshake flags.")

# 4. Inspect Final Answer
print("\n=== ANSWER ===")
print(response.answer)
print("Confidence:", response.confidence)

# 5. Inspect Step-by-Step Observable Trajectory
print("\n=== REASONING TRAJECTORY ===")
for step in response.trajectory:
    print(f"\n>> Step {step.step_num}: Action={step.action} (Latency: {step.took_ms}ms)")
    print(f"   Thought: {step.thought}")
    print(f"   Observation: {step.observation[:120]}...")
```
