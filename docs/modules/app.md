# Module: `polyrag.app` & `polyrag.service` 🏛️

The `polyrag.app` module provides `PolyRAG`, the primary user-facing Application Context and Pipeline Factory for the entire framework.

The `polyrag.service` module provides `RAGService` and `AgenticRAGService`, which are backward-compatibility aliases to `PolyRAG` and `AgenticRAG`.

---

## 1. What is `PolyRAG`?

`PolyRAG` acts as the single entry point for:
1. **Configuring Environment & Storage**: Sets up your LLM, embeddings, vector database, and chunker.
2. **Shared Ingestion**: Ingests files, raw text, directories, and LangChain Document Loaders once into the underlying vector store.
3. **Pipeline Factory**: Spawns specialized pipelines (`NaiveRAG`, `AdvancedRAG`, `AgenticRAG`, `ReActAgent`) that operate on the populated vector store.
4. **Convenience Execution**: Directly answers questions using the default baseline strategy via `rag.query(...)`.

---

## 2. Construction Methods

### A. Factory Method: `PolyRAG.create(...)` *(Recommended)*
Explicitly inject your choice of native LangChain components or PolyRAG interfaces:
```python
from polyrag.app import PolyRAG
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(collection_name="kb_docs", embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini")

rag = PolyRAG.create(
    vector_store=vector_store,
    embedding_model=embeddings,
    chat_model=llm,
)
```

### B. Environment Auto-Configuration: `PolyRAG.from_env(...)`
Automatically reads `.env` and environment variables (`OPENAI_API_KEY`, `CHROMA_PERSIST_DIR`, `MODEL_NAME`):
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
# 1. Ingest raw text
rag.ingest_text(
    text="OAuth 2.0 uses access tokens and refresh tokens...",
    source="oauth_spec.md",
)

# 2. Ingest a single file
rag.ingest_file("docs/rfc9000.txt")

# 3. Ingest an entire directory of documentation
rag.ingest_directory("knowledge_base/", glob_pattern="**/*.md")

# 4. Ingest from 150+ LangChain Document Loaders (PDF, DOCX, CSV, Web)
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/annual_report.pdf")
rag.ingest_langchain_loader(loader, metadata={"department": "finance"})

# 5. Ingest arbitrary document iterables or generators
rag.ingest_documents([
    {"text": "Chunk text 1", "source": "doc1.txt"},
    {"text": "Chunk text 2", "source": "doc2.txt"},
])
```

---

## 4. Spawning Specialized Pipelines

Once data is ingested, you can instantiate any specialized RAG strategy sharing the same index:

```python
# 1. 1-Shot Naive RAG
naive = rag.create_naive_rag()
res1 = naive.query("What is OAuth 2.0?")

# 2. Advanced RAG with Multi-Query Expansion and RRF
advanced = rag.create_advanced_rag(
    top_k=5,
    num_expanded_queries=3,
    min_relevance_score=0.1,
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
react = rag.create_react_agent(max_steps=5)
res4 = react.query("Investigate authentication errors in staging.")
```

---

## 5. Backward Compatibility: `RAGService` & `AgenticRAGService`

For existing code referencing `RAGService`, it is preserved as an alias for `PolyRAG`:

```python
from polyrag.service import RAGService, AgenticRAGService

service = RAGService.from_env()
service.ingest("System architecture details...", source="arch.md")
response = service.query("Explain the architecture")
```
