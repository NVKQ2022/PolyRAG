"""Unit tests for the PolyRAG application context and pipeline factory."""

from pathlib import Path
from typing import Any
import tempfile
import pytest

from polyrag import (
    AdvancedRAG,
    AgenticRAG,
    AgenticRAGService,
    Container,
    NaiveRAG,
    PolyRAG,
    RAGResponse,
    RAGService,
    ReActAgent,
)
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.vector_stores.memory import InMemoryVectorStore


class MockEmbedding(BaseEmbeddingModel):
    @property
    def dim(self) -> int:
        return 4

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class MockLLM(BaseLLMClient):
    @property
    def model_name(self) -> str:
        return "mock-llm"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        if "optimizer" in prompt:
            return "QUIC transport\nQUIC RFC 9000\nQUIC connection"
        return "Grounded synthesis response based on retrieved facts."

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        if "retrieval planner" in prompt:
            return {"retrieval_needed": True, "query": "QUIC", "reason": "Query required"}
        if "reflection and answering" in prompt:
            return {
                "enough": True,
                "confidence": 0.9,
                "reason": "Satisfied",
                "missing_gaps": [],
                "next_query": "",
                "answer": "Grounded answer from reflection.",
            }
        return {}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "Mock chat response"


def test_polyrag_create_and_ingest():
    """Verify PolyRAG context initialization and shared document ingestion."""
    emb = MockEmbedding()
    vdb = InMemoryVectorStore()
    llm = MockLLM()

    app = PolyRAG(
        embedding_model=emb,
        vector_store=vdb,
        llm_client=llm,
    )

    # Ingest text
    app.ingest("RFC 793 defines the Transmission Control Protocol (TCP).", source="rfc793.txt")
    assert vdb.count() == 1

    # Ingest file
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write("RFC 768 defines the User Datagram Protocol (UDP).")
        temp_path = f.name

    try:
        app.ingest_file(temp_path)
        assert vdb.count() == 2
    finally:
        Path(temp_path).unlink(missing_ok=True)

    # Retrieve from shared store
    results = app.retrieve("What is TCP?", top_k=1)
    assert len(results) == 1
    assert "Transmission Control Protocol" in results[0]["document"]["text"]


def test_polyrag_pipeline_factory():
    """Verify PolyRAG manufactures all RAG pipeline types from the configured setup."""
    emb = MockEmbedding()
    vdb = InMemoryVectorStore()
    llm = MockLLM()

    app = PolyRAG(
        embedding_model=emb,
        vector_store=vdb,
        llm_client=llm,
    )

    app.ingest("RFC 9000 defines QUIC transport.", source="quic.txt")

    # 1. Naive RAG
    naive = app.create_naive_rag()
    assert isinstance(naive, NaiveRAG)
    res_naive = naive.query("What is QUIC?", top_k=1)
    assert isinstance(res_naive, RAGResponse)
    assert res_naive.answer != ""

    # 2. Advanced RAG
    advanced = app.create_advanced_rag(num_expanded_queries=2, top_k=2)
    assert isinstance(advanced, AdvancedRAG)
    res_adv = advanced.query("What is QUIC?", top_k=1)
    assert isinstance(res_adv, RAGResponse)

    # 3. Agentic RAG
    agentic = app.create_agentic_rag(max_rounds=2)
    assert isinstance(agentic, AgenticRAG)
    res_agentic = agentic.query("What is QUIC?")
    assert isinstance(res_agentic, RAGResponse)

    # 4. ReAct Agent
    react = app.create_react_agent(max_steps=3)
    assert isinstance(react, ReActAgent)


def test_polyrag_query_shortcut():
    """Verify PolyRAG.query() provides a clean 1-line execution shortcut."""
    app = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    app.ingest("RFC 1035 defines Domain Names.", source="rfc1035.txt")

    response = app.query("What is DNS?", top_k=1)
    assert isinstance(response, RAGResponse)
    assert response.confidence > 0.0


def test_polyrag_from_container():
    """Verify PolyRAG can be constructed from a Dependency Injection Container."""
    container = Container.create(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )

    app = container.build_app()
    assert isinstance(app, PolyRAG)
    app.ingest("Container test document.")
    assert app.vector_store.count() == 1


def test_rag_service_inherits_polyrag():
    """Verify backward compatibility: RAGService is a subclass and instance of PolyRAG."""
    service = RAGService(
        client=MockLLM(),
        embedding_service=MockEmbedding(),
        vector_db=InMemoryVectorStore(),
    )

    assert isinstance(service, PolyRAG)
    service.ingest("Backward compatibility doc.", source="compat.txt")
    assert service.vector_store.count() == 1

    # Verify both new and legacy factory methods work
    adv = service.create_advanced_rag()
    assert isinstance(adv, AdvancedRAG)

    alias_adv = service.as_advanced()
    assert isinstance(alias_adv, AdvancedRAG)
