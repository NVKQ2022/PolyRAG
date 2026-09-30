"""Unit tests verifying the RAG class hierarchy (BaseRAG -> NaiveRAG, AdvancedRAG, AgenticRAG, ReActAgent)."""

from typing import Any
import pytest

from polyrag.core.interfaces import BaseEmbeddingModel, BaseLLMClient
from polyrag.core.models import RAGResponse
from polyrag.pipelines.advanced import AdvancedRAG
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.base import BaseRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent, ReActRAG
from polyrag.service import RAGService
from polyrag.vector_stores.memory import InMemoryVectorStore


class MockEmbedding(BaseEmbeddingModel):
    @property
    def dim(self) -> int:
        return 3

    def embed_text(self, text: str) -> list[float]:
        t = text.lower()
        return [
            1.0 if "dns" in t else 0.1,
            1.0 if "tcp" in t else 0.1,
            1.0 if "http" in t else 0.1,
        ]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class MockLLM(BaseLLMClient):
    @property
    def model_name(self) -> str:
        return "mock-llm"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        if "optimizer" in prompt:
            # Query expansion
            return "DNS domain resolution\nDomain Name System architecture\nDNS RFC specification"
        return "Grounded response citing [rfc1035.txt#0]."

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        if "retrieval planner" in prompt:
            return {"retrieval_needed": True, "query": "DNS RFC", "reason": "Technical query"}
        if "reflection and answering" in prompt:
            return {
                "enough": True,
                "confidence": 0.95,
                "reason": "Found facts",
                "missing_gaps": [],
                "next_query": "",
                "answer": "Grounded answer from reflection [rfc1035.txt#0].",
            }
        return {}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "Mock chat response"


def test_rag_class_hierarchy_inheritance():
    """Verify that all RAG pipelines correctly inherit from BaseRAG."""
    assert issubclass(NaiveRAG, BaseRAG)
    assert issubclass(AdvancedRAG, BaseRAG)
    assert issubclass(AgenticRAG, BaseRAG)
    assert issubclass(ReActAgent, BaseRAG)
    assert issubclass(ReActRAG, BaseRAG)


def test_base_rag_shared_ingestion_and_retrieval():
    """Verify shared BaseRAG methods work identically on derived classes."""
    emb = MockEmbedding()
    vdb = InMemoryVectorStore()
    llm = MockLLM()

    naive = NaiveRAG(embedding_model=emb, vector_store=vdb, llm_client=llm)
    docs = naive.ingest_text("RFC 1035 Domain Name System", source="rfc1035.txt")
    assert len(docs) == 1
    assert vdb.count() == 1

    # Shared retrieve method
    results = naive.retrieve("DNS", top_k=1)
    assert len(results) == 1

    # Shared format_context method
    ctx = naive.format_context(results)
    assert "rfc1035.txt#0" in ctx


def test_advanced_rag_pipeline():
    """Verify AdvancedRAG query expansion, RRF re-ranking, and execution."""
    emb = MockEmbedding()
    vdb = InMemoryVectorStore()
    llm = MockLLM()

    vdb.add_documents(
        vectors=[emb.embed_text("DNS domain resolution details")],
        documents=[{"source": "rfc1035.txt", "chunk_id": 0, "text": "DNS details"}],
    )

    advanced = AdvancedRAG(
        embedding_model=emb,
        vector_store=vdb,
        llm_client=llm,
        num_expanded_queries=3,
        top_k=2,
    )

    # Test query expansion
    expanded = advanced.expand_query("Explain DNS")
    assert len(expanded) >= 2

    # Test RRF fusion
    runs = [
        [{"document": {"source": "doc1.txt", "chunk_id": 0}, "score": 0.9}],
        [{"document": {"source": "doc1.txt", "chunk_id": 0}, "score": 0.8}],
    ]
    fused = advanced.reciprocal_rank_fusion(runs)
    assert len(fused) == 1
    assert "rrf_score" in fused[0]

    # Test execution
    response = advanced.execute("Explain DNS protocol", top_k=1)
    assert isinstance(response, RAGResponse)
    assert response.confidence >= 0.8
    assert "Expanded" in response.reasoning_summary


def test_rag_service_as_advanced():
    """Verify RAGService.as_advanced() converts into an AdvancedRAG instance."""
    emb = MockEmbedding()
    vdb = InMemoryVectorStore()
    llm = MockLLM()

    service = RAGService(
        client=llm,
        embedding_service=emb,
        vector_db=vdb,
    )

    service.ingest("RFC 793 Transmission Control Protocol", source="rfc793.txt")

    advanced = service.as_advanced(num_expanded_queries=2)
    assert isinstance(advanced, AdvancedRAG)
    assert isinstance(advanced, BaseRAG)

    res = advanced.query("What is TCP?", top_k=1)
    assert isinstance(res, RAGResponse)
