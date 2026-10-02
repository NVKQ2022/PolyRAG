# Custom Pipeline Configuration

PolyRAG is architected around the **Dependency Inversion Principle** and native LangChain compatibility. Every component adheres to standard interfaces defined in `polyrag.core.interfaces` and `langchain_core`. You can swap any component or pass any LangChain partner package without modifying core pipelines.

---

## 1. Chunkers (`BaseChunker` / `TextSplitter`)

### `RecursiveCharacterChunker` (Recommended)
Recursively splits text using natural hierarchy (paragraphs, newlines, sentences, spaces):

```python
from polyrag import RecursiveCharacterChunker

chunker = RecursiveCharacterChunker(
    chunk_size=550,           # Max chunk character length
    chunk_overlap=50,         # Overlap between consecutive chunks
    separators=["\n\n", "\n", ". ", " ", ""],
    drop_empty=True,          # Discard whitespace chunks
)
```

### `FixedSizeChunker`
Fixed-window slicing with overlap:

```python
from polyrag import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=400,
    chunk_overlap=40,
    drop_empty=True,
)
```

---

## 2. Embeddings (`BaseEmbeddingModel` / `Embeddings`)

PolyRAG accepts any LangChain `Embeddings` model or custom port implementation:

### Local HuggingFace / Transformers (`langchain-huggingface`)
```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### OpenAI Embeddings (`langchain-openai`)
```python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

### Zero-Setup Dev / Test Mode
```python
from polyrag.embeddings import resolve_embedding_model

# Synthetic deterministic embeddings with no external network calls
embedding_model = resolve_embedding_model("fake", size=384)
```

---

## 3. Vector Stores (`BaseVectorStore` / `VectorStore`)

### `InMemoryVectorStore` (Zero-Dependency)
Thread-safe in-memory vector store with cosine similarity ranking. Perfect for testing and ephemeral tasks:

```python
from polyrag import InMemoryVectorStore

vector_store = InMemoryVectorStore()
```

### Chroma (`langchain-chroma`)
```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="production_docs",
    embedding_function=embedding_model,
    persist_directory="./chroma_db",
)
```

### Milvus & Milvus Lite (`langchain-milvus`)
```python
from langchain_milvus import Milvus

vector_store = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri": "./milvus_demo.db"},
    collection_name="production_docs",
)
```

---

## 4. Chat Models & LLMs (`BaseLLMClient` / `BaseChatModel`)

Pass any standard LangChain ChatModel or use string resolution:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)
```

---

## 5. Assembling Any Pipeline in the Hierarchy

Once components are configured, pass them to any pipeline deriving from `BaseRAG`:

```python
from polyrag import NaiveRAG, AdvancedRAG, AgenticRAG, ReActAgent

# Standard Naive RAG
naive = NaiveRAG(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm_client=llm,
)

# Advanced RAG (Multi-Query Expansion & RRF)
advanced = AdvancedRAG(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm_client=llm,
    num_expanded_queries=3,
    top_k=5,
)

# Agentic RAG (Autonomous Planning & Fused Reflection)
agentic = AgenticRAG(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm_client=llm,
    top_k=3,
    max_rounds=2,
    verbose=True,
)

# ReAct Agent (Dynamic Tool Calling & Reasoning Loop)
react = ReActAgent(
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm_client=llm,
    max_steps=4,
)
```

---

## 6. Zero Custom Adapters Needed

Because PolyRAG natively implements LangChain interfaces:
- Any vector database supported by LangChain (Pinecone, Qdrant, PGVector, FAISS, Weaviate) works out-of-the-box.
- Any LLM supported by LangChain (Claude, Gemini, Ollama, Groq, Mistral) works out-of-the-box.
- No adapter shims or custom driver classes required!
