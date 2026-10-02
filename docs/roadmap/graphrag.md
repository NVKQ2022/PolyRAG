# Roadmap: GraphRAG & Multi-Paradigm Retrieval 🗺️

PolyRAG is engineered to expand beyond vector-only indexing into **GraphRAG** and **Hybrid Retrieval**, unifying unstructured semantic embeddings with structured relational knowledge.

---

## 1. Why GraphRAG?

Vector databases excel at local semantic lookup (*"find passages similar to this phrase"*), but fail on two core query archetypes:
1. **Global Corpus Summaries**: Questions like *"What are the overarching protocol design patterns across all 15 RFC specifications?"* require synthesizing the whole corpus, not just 5 retrieved chunks.
2. **Multi-Hop Relational Traversal**: When the answer requires following transitive relationships (*Entity A connects to B, which depends on C*), pure vector embeddings often miss intermediary link nodes.

GraphRAG extracts structured entities, discovers latent community clusters, generates hierarchical summaries, and provides graph-guided context.

---

## 2. GraphRAG Pipeline Architecture

```
[Raw Ingested Documents]
           │
           ▼ (1. Entity & Relationship Extraction via LLM)
   [Nodes & Edges Extracted]
           │
           ▼ (2. Knowledge Graph Indexing)
   [Graph Store: NetworkX / Neo4j]
           │
           ▼ (3. Community Detection: Leiden / Louvain)
   [Hierarchical Graph Communities]
           │
           ▼ (4. Community Summarization via LLM)
   [Cluster Reports & Vector Embeddings]
           │
           ▼ (5. Global / Local Graph Retrieval)
   [Graph-Grounded Answer Synthesis]
```

---

## 3. Integration into the `BaseRAG` Hierarchy

`GraphRAG` will integrate seamlessly into PolyRAG's class hierarchy:

```
                       ┌─────────────────────────┐
                       │         BaseRAG         │
                       └────────────┬────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
  │   NaiveRAG   │           │ AdvancedRAG  │           │  AgenticRAG  │
  └──────────────┘           └──────────────┘           └──────────────┘
                                    │
                                    ▼
                             ┌──────────────┐
                             │   GraphRAG   │
                             │ (Knowledge)  │
                             └──────────────┘
```

### Planned Syntax

```python
from polyrag import GraphRAG, InMemoryVectorStore, ChatOpenAI

# Planned future usage:
graph_rag = GraphRAG(
    vector_store=InMemoryVectorStore(),
    llm_client=ChatOpenAI(model="gpt-4o"),
)

# Global corpus-level reasoning
response = graph_rag.query_global(
    "What are the recurring error handling principles across all networking protocols?"
)

# Local relational entity traversal
response = graph_rag.query_local(
    "How does TLS 1.3 key negotiation integrate into QUIC connection migration?"
)
```

---

## 4. Planned Interfaces (`polyrag.core.interfaces`)

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseGraphStore(ABC):
    """Abstract interface for knowledge graph storage and community querying."""

    @abstractmethod
    def add_entity(self, name: str, entity_type: str, description: str) -> None: ...

    @abstractmethod
    def add_relation(self, source: str, target: str, relationship: str, weight: float = 1.0) -> None: ...

    @abstractmethod
    def query_subgraph(self, entity_names: list[str], max_depth: int = 2) -> dict[str, Any]: ...

    @abstractmethod
    def detect_communities(self) -> list[dict[str, Any]]: ...
```
