# Module: `polyrag.app` & `polyrag.service` 🏛️

The `polyrag.app` module provides `PolyRAG`, the primary user-facing Application Context and Pipeline Factory for the entire framework.

The `polyrag.service` module provides `RAGService` and `AgenticRAGService`, which inherit directly from `PolyRAG` to ensure 100% backward compatibility with legacy service code.

---

## 1. What is `PolyRAG`?

`PolyRAG` acts as the single entry point for:
1. **Configuring Environment & Storage**: Initializes your LLM, embeddings, vector database, and chunker.
2. **Shared Ingestion**: Ingests files, raw text, and directories once into the underlying vector store.
3. **Pipeline Factory**: Spawns specialized pipelines (`NaiveRAG`, `AdvancedRAG`, `AgenticRAG`, `ReActAgent`) that operate on the shared populated vector store.
4. **Convenience Execution**: Directly answers questions using the default baseline strategy via `rag.query(...)`.

---

## 2. Construction Methods

### A. Factory Method: `PolyRAG.create(...)` *(Recommended)*
Explicitly inject your choice of components:
```python
from polyrag.app import PolyRAG
from polyrag.vector_stores import MilvusVectorStore
from polyrag.embeddings import SentenceTransformerEmbedding
from polyrag.llms import OpenAILLM

rag = PolyRAG.create(
    vector_store=MilvusVectorStore(uri="./milvus_demo.db"),
    embedding_model=SentenceTransformerEmbedding("all-MiniLM-L6-v2"),
    llm_client=OpenAILLM(model_name="gpt-4o-mini"),
)
```

### B. Environment Auto-Configuration: `PolyRAG.from_env(...)`
Automatically reads `.env` and environment variables (`OPENAI_API_KEY`, `CHROMA_PERSIST_DIR`, `EMBEDDING_MODEL`):
```python
rag = PolyRAG.from_env()
```

### C. DI Container Binding: `PolyRAG.from_container(container)`
Constructs `PolyRAG` wired directly to an existing `Container`:
```python
rag = PolyRAG.from_container(container)
```

---

## 3. Shared Document Ingestion

Documents ingested into `PolyRAG` are chunked, embedded, and indexed once into the active vector store:

```python
# Ingest raw text
rag.ingest_text(
    text="OAuth 2.0 uses access tokens and refresh tokens...",
    source="oauth_spec.md",
)

# Ingest a single file
rag.ingest_file("docs/handbook.pdf")

# Ingest an entire directory of documentation
rag.ingest_directory("knowledge_base/", glob_pattern="**/*.md")
```

---

## 4. Spawning Specialized Pipelines

Once data is ingested, you can instantiate any specialized RAG strategy sharing the same index:

```python
# 1. 1-Shot Naive RAG
naive = rag.create_naive_rag()
res1 = naive.query("What is OAuth 2.0?")

# 2. Advanced RAG with Query Expansion and RRF
advanced = rag.create_advanced_rag(
    top_k=5,
    num_expanded_queries=3,
    min_relevance_score=0.3,
)
res2 = advanced.query("How do refresh tokens interact with token rotation?")

# 3. Autonomous Agentic RAG with Self-Reflection
agentic = rag.create_agentic_rag(
    top_k=3,
    max_rounds=2,
    verbose=True,
)
res3 = agentic.query("Compare OAuth 2.0 PKCE with client secrets.")

# 4. Tool-Driven ReAct Agent
agent = rag.create_react_agent(tools=[...], max_steps=5)
res4 = agent.run("Investigate authentication errors in staging.")
```

---

## 5. `RAGService` & `AgenticRAGService` (`polyrag.service`)

For existing applications using `RAGService`, `RAGService` subclasses `PolyRAG`:

```python
from polyrag.service import RAGService, AgenticRAGService

# Drop-in compatible with all PolyRAG factory methods and query pipelines
service = RAGService.from_env()
service.ingest_text("System architecture details...", source="arch.md")
response = service.query("Explain the architecture")
```

| Class | Base Class | Recommendation |
| :--- | :--- | :--- |
| **`PolyRAG`** | Domain Object | **Use for new projects.** Primary application context and pipeline factory. |
| **`RAGService`** | `PolyRAG` | **Use for legacy code.** Preserved for 100% backward compatibility. |
| **`AgenticRAGService`**| `RAGService` | Legacy agentic entry point. |
