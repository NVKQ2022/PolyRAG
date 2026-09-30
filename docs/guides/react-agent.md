# ReAct Agent Guide 🤖

The **ReActAgent** pipeline implements the **Reasoning + Acting** pattern (Yao et al., 2022). It operates in a strict, observable cycle:
1. **Thought**: The agent reasons about current facts and identifies missing information.
2. **Action**: The agent selects and invokes a tool with specific parameters.
3. **Observation**: The system executes the tool and feeds the result back into the prompt.
4. **Final Answer**: Once sufficient evidence is assembled, the agent synthesizes the grounded answer.

---

## Architecture

```
[User Question]
       │
       ▼
┌───────────────────────────────────────────────┐
│              ReAct Execution Loop             │
│                                               │
│  1. Thought: "I need to look up RFC 9000"     │
│  2. Action:  search({"query": "QUIC 9000"})   │
│  3. Observation: [rfc9000.txt#12] retrieved   │
│                                               │
│  (Repeats up to max_steps or final_answer)   │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
            [AgentResponse with Trajectory]
```

---

## Built-In Tools

| Tool Name | Action Input | Purpose |
| :--- | :--- | :--- |
| `search` | `{"query": str, "top_k": int}` | Semantic similarity search in vector knowledge base |
| `list_docs` | `{}` | Overview of total indexed chunks and sample metadata |
| `final_answer` | `{"answer": str, "confidence": float}` | Concludes reasoning with a grounded answer citing sources |

---

## Usage Example

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

# 2. Populate knowledge base
vdb.add_documents(
    vectors=[emb.embed_text("QUIC relies on UDP to provide stream multiplexing.")],
    documents=[{"source": "rfc9000.txt", "chunk_id": 1, "text": "QUIC relies on UDP."}],
)

# 3. Create Agent
agent = ReActAgent(
    llm_client=llm,
    embedding_model=emb,
    vector_store=vdb,
    max_steps=4,
    default_top_k=3,
    verbose=True,
)

# 4. Query Agent
response = agent.query(
    "What transport protocol does QUIC build upon according to RFC 9000?"
)

# 5. Inspect Trajectory
print("\n=== FINAL ANSWER ===")
print(response.answer)

print("\n=== STEP-BY-STEP TRAJECTORY ===")
for step in response.trajectory:
    print(f"\n>> Step {step.step_num}: Action={step.action} ({step.took_ms}ms)")
    print(f"   Thought: {step.thought}")
    print(f"   Observation: {step.observation[:120]}...")
```
