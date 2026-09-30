# Roadmap: GraphRAG & Multi-Paradigm Retrieval 🗺️

PolyRAG is designed to expand beyond vector-only search into **GraphRAG** and **Hybrid Retrieval**, bridging unstructured text embeddings with structured relational knowledge.

---

## The Motivation for GraphRAG

Standard vector retrieval excels at localized semantic similarity ("find text similar to this query"), but struggles with:
1. **Holistic / Global Questions**: Questions like *"What are the primary recurring architectural challenges across all protocols?"* require summarizing an entire corpus, not just 5 chunks.
2. **Multi-Hop Relational Links**: Traversing entity relationships (e.g., *RFC 9000 uses TLS 1.3 (RFC 8446), which replaces RSA key exchange*).

GraphRAG combines Knowledge Graph extraction with LLM community summarization to answer both local and global queries.

---

## Planned Architecture

```
[Raw Documents]
       │
       ▼ (Entity & Relationship Extraction)
[Entities, Claims, & Relations]
       │
       ▼ (Graph Construction)
[Knowledge Graph (NetworkX / Neo4j)]
       │
       ▼ (Leiden / Louvain Community Detection)
[Hierarchical Graph Communities]
       │
       ▼ (LLM Community Summarization)
[Community Summaries & Vector Indexing]
```

---

## Planned Extension Interfaces

To maintain PolyRAG's plug-and-play philosophy, GraphRAG will be introduced with abstract ports:

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseKnowledgeGraph(ABC):
    """Abstract interface for knowledge graph stores."""

    @abstractmethod
    def add_entity(self, name: str, entity_type: str, description: str) -> None: ...

    @abstractmethod
    def add_relation(self, source: str, target: str, relationship: str, weight: float = 1.0) -> None: ...

    @abstractmethod
    def query_subgraph(self, entity_names: list[str], max_depth: int = 2) -> dict[str, Any]: ...
```

---

## Hybrid Search (Vectors + Knowledge Graph)

The upcoming `HybridRAG` pipeline will fuse results using Reciprocal Rank Fusion (RRF):

```python
# Planned future syntax:
from polyrag import HybridRAG, ChromaVectorStore, NetworkXGraphStore

rag = HybridRAG(
    vector_store=ChromaVectorStore(persist_path="./chroma_db"),
    graph_store=NetworkXGraphStore(persist_path="./graph.json"),
    llm_client=OpenAILLM(model_name="gpt-4o"),
)

# Combines semantic vector chunks + relational subgraph context
response = rag.query("How do TLS 1.3 and QUIC interconnect?")
```
