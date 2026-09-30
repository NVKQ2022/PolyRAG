# The RAG Class Hierarchy 🏛️

PolyRAG is structured around an object-oriented class hierarchy rooted in `BaseRAG`. This guarantees that all pipelines share common ingestion, vector storage, and context retrieval primitives while allowing specialized reasoning workflows.

---

## 1. Class Hierarchy Diagram

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
                                                        │ (ReActRAG)   │
                                                        └──────────────┘
```

---

## 2. Common Inherited Primitives (`BaseRAG`)

All pipelines inherit the following methods from `BaseRAG`:

| Method | Description |
| :--- | :--- |
| `ingest_text(text, source, metadata)` | Chunks, embeds, and indexes raw text into the vector store. |
| `ingest_file(file_path, metadata)` | Reads a file from disk and indexes its contents. |
| `ingest_directory(dir_path, glob_pattern, metadata)` | Recursively indexes all matching files in a directory. |
| `retrieve(query, top_k)` | Computes query vector and searches nearest neighbors. |
| `format_context(search_results)` | Assembles retrieved chunks into structured context blocks. |
| `execute(question, **kwargs)` | *(Abstract)* Implemented by each derived class for custom answering logic. |
| `query(question, **kwargs)` | Convenience alias that forwards to `execute()`. |

---

## 3. Comparison of Derived Pipelines

| Feature | `NaiveRAG` | `AdvancedRAG` | `AgenticRAG` | `ReActAgent` |
| :--- | :--- | :--- | :--- | :--- |
| **Strategy** | 1-Shot Retrieve-then-Read | Pre/Post-Retrieval Optimization | Autonomous Cognitive Feedback | Tool-driven ReAct Loop |
| **Query Expansion** | ❌ None | ✅ Multi-Query Expansion | ✅ Keyword-dense Rewriting | ✅ Iterative Tool Inputs |
| **Re-Ranking** | ❌ Raw similarity | ✅ Reciprocal Rank Fusion (RRF) | ✅ Deduplication across rounds | ✅ Trajectory Deduplication |
| **Reflection / Self-Check** | ❌ None | ❌ None | ✅ Fused Reflection | ✅ Thought Steps |
| **Conversational Bypass** | ❌ Always retrieves | ❌ Always retrieves | ✅ Direct Answer for greetings | ❌ Tool execution |
| **LLM Calls** | 1 | $1 + \text{queries}$ | $1 \sim 3$ (Fused) | $1 \sim \text{max\_steps}$ |
| **Best For** | Simple factual lookups | Multi-perspective search | Complex multi-hop synthesis | Dynamic multi-tool reasoning |

---

## 4. Polymorphic Pipeline Usage

Because every pipeline inherits from `BaseRAG`, your application can accept `BaseRAG` as a type annotation and swap strategies without code changes:

```python
from polyrag import BaseRAG, NaiveRAG, AdvancedRAG, AgenticRAG

def answer_user_query(pipeline: BaseRAG, question: str) -> str:
    """Polymorphic helper working with ANY pipeline in the hierarchy."""
    response = pipeline.query(question)
    return response.answer
```

---

## 5. Manufacturing Pipelines via `PolyRAG`

Instead of wiring dependencies manually for each pipeline, use [`PolyRAG`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/app.py) as the central setup orchestrator and factory:

```python
from polyrag import PolyRAG

# 1. Setup models and vector storage once
app = PolyRAG.from_env()

# 2. Ingest shared knowledge
app.ingest_file("handbook.pdf")

# 3. Manufacture whichever pipeline is needed
naive = app.create_naive_rag()
advanced = app.create_advanced_rag(num_expanded_queries=3)
agentic = app.create_agentic_rag(max_rounds=2)
react = app.create_react_agent(max_steps=4)

# All manufactured objects inherit from BaseRAG!
assert isinstance(naive, BaseRAG)
assert isinstance(advanced, BaseRAG)
assert isinstance(agentic, BaseRAG)
assert isinstance(react, BaseRAG)
```
