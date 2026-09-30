"""Unit tests for the modular polyrag package."""

from pathlib import Path
import tempfile
from typing import Any
import pytest

from polyrag.chunkers.fixed_size import FixedSizeChunker
from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.core.interfaces import BaseEmbeddingModel, BaseLLMClient
from polyrag.core.models import (
    AgentAction,
    AgentResponse,
    AgentStep,
    Chunk,
    Document,
    RAGResponse,
    SearchResult,
)
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent
from polyrag.service import AgenticRAGService, RAGService
from polyrag.vector_stores.memory import InMemoryVectorStore


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic embedding model for fast unit testing."""

    @property
    def dim(self) -> int:
        return 4

    def embed_text(self, text: str) -> list[float]:
        # Simple deterministic 4D vector based on text keywords
        text_lower = text.lower()
        v0 = 1.0 if "dns" in text_lower or "1035" in text_lower else 0.1
        v1 = 1.0 if "tcp" in text_lower or "793" in text_lower else 0.1
        v2 = 1.0 if "http" in text_lower or "quic" in text_lower else 0.1
        v3 = float(len(text) % 10) / 10.0
        return [v0, v1, v2, v3]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class MockLLMClient(BaseLLMClient):
    """Deterministic LLM mock responding to prompt cues."""

    def __init__(self, model_name: str = "mock-gpt") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, prompt: str, **kwargs: Any) -> str:
        if "politely and concisely" in prompt or "conversational" in prompt:
            return "Hello! How can I assist you with network RFCs today?"
        if "ReAct" in prompt or "Action" in prompt:
            if "Step: 1" in prompt:
                return '{"thought": "I will search for DNS protocol details.", "action": "search", "action_input": {"query": "DNS RFC 1035", "top_k": 2}}'
            return '{"thought": "I have collected sufficient evidence.", "action": "final_answer", "action_input": {"answer": "DNS is defined in RFC 1035 [rfc1035.txt#0].", "confidence": 0.98}}'
        return "RFC 1035 grounded answer based on evidence [rfc1035.txt#0]."

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        if "retrieval planner" in prompt:
            question_section = prompt.split("Question:")[-1].split("Decide whether")[0].lower()
            if any(w in question_section.split() for w in ("hello", "hi", "hey")):
                return {
                    "retrieval_needed": False,
                    "query": "",
                    "reason": "Greeting does not require retrieval.",
                }
            return {
                "retrieval_needed": True,
                "query": "DNS RFC 1035 specification",
                "reason": "Technical networking question.",
            }
        if "reflection and answering" in prompt:
            return {
                "enough": True,
                "confidence": 0.95,
                "reason": "Found clear specification in chunk.",
                "missing_gaps": [],
                "next_query": "",
                "answer": "DNS defines domain namespace in RFC 1035 [rfc1035.txt#0].",
            }
        return {}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self.complete(messages[-1]["content"])


# =====================================================================
# Tests: Chunkers
# =====================================================================

def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
    text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda"
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(len(c) <= 25 for c in chunks)


def test_recursive_character_chunker():
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    text = (
        "RFC 1035 describes the Domain Name System.\n\n"
        "It provides hierarchical name resolution across the internet.\n\n"
        "Each domain name consists of labels separated by dots."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    assert "RFC 1035" in chunks[0]


# =====================================================================
# Tests: InMemoryVectorStore
# =====================================================================

def test_in_memory_vector_store():
    store = InMemoryVectorStore()
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    docs = [
        {"source": "dns.txt", "chunk_id": 0, "text": "DNS chunk"},
        {"source": "tcp.txt", "chunk_id": 0, "text": "TCP chunk"},
        {"source": "http.txt", "chunk_id": 0, "text": "HTTP chunk"},
    ]
    store.add_documents(vectors=vectors, documents=docs)
    assert store.count() == 3

    # Query close to DNS
    results = store.search(query_vector=[0.9, 0.1, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["document"]["source"] == "dns.txt"
    assert results[0]["score"] > results[1]["score"]

    # Peek and clear
    peek_res = store.peek(limit=2)
    assert len(peek_res["documents"]) == 2
    store.clear()
    assert store.count() == 0


# =====================================================================
# Tests: NaiveRAG Pipeline
# =====================================================================

def test_naive_rag_pipeline():
    emb = MockEmbeddingModel()
    vdb = InMemoryVectorStore()
    llm = MockLLMClient()

    rag = NaiveRAG(embedding_model=emb, vector_store=vdb, llm_client=llm)

    # Ingest text
    docs = rag.ingest_text(
        text="RFC 1035 specifies domain name system structure.",
        source="rfc1035.txt",
    )
    assert len(docs) >= 1
    assert vdb.count() >= 1

    # Ingest file
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write("RFC 793 defines the Transmission Control Protocol.")
        tmp_path = f.name

    try:
        f_docs = rag.ingest_file(tmp_path)
        assert len(f_docs) >= 1
        assert vdb.count() >= 2
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Retrieve
    results = rag.retrieve("DNS domain", top_k=1)
    assert len(results) == 1
    assert "rfc1035" in results[0]["document"]["source"]

    # Execute
    res = rag.execute("What is RFC 1035?", top_k=1)
    assert isinstance(res, RAGResponse)
    assert "rfc1035" in res.context
    assert res.answer != ""
    assert res["answer"] == res.answer  # test __getitem__


# =====================================================================
# Tests: AgenticRAG Pipeline
# =====================================================================

def test_agentic_rag_direct_answer():
    emb = MockEmbeddingModel()
    vdb = InMemoryVectorStore()
    llm = MockLLMClient()

    agentic = AgenticRAG(
        embedding_model=emb,
        vector_store=vdb,
        llm_client=llm,
    )

    # Conversational greeting -> direct answer
    response = agentic.query("Hello there!")
    assert isinstance(response, RAGResponse)
    assert "Hello" in response.answer
    assert response.confidence == 1.0
    assert len(response.sources) == 0


def test_agentic_rag_multiround():
    emb = MockEmbeddingModel()
    vdb = InMemoryVectorStore()
    llm = MockLLMClient()

    # Pre-populate store
    vdb.add_documents(
        vectors=[emb.embed_text("RFC 1035 domain name system architecture")],
        documents=[{"source": "rfc1035.txt", "chunk_id": 0, "text": "DNS specification details."}],
    )

    agentic = AgenticRAG(
        embedding_model=emb,
        vector_store=vdb,
        llm_client=llm,
        top_k=2,
        max_rounds=2,
    )

    response = agentic.query("Explain DNS specification in RFC 1035")
    assert isinstance(response, RAGResponse)
    assert "RFC 1035" in response.answer
    assert len(response.sources) >= 1
    assert response.confidence >= 0.9
    assert len(response.agent_log) >= 2


# =====================================================================
# Tests: ReActAgent Pipeline
# =====================================================================

def test_react_agent_execution():
    emb = MockEmbeddingModel()
    vdb = InMemoryVectorStore()
    llm = MockLLMClient()

    vdb.add_documents(
        vectors=[emb.embed_text("RFC 1035 DNS specification")],
        documents=[{"source": "rfc1035.txt", "chunk_id": 0, "text": "DNS specification details."}],
    )

    agent = ReActAgent(
        llm_client=llm,
        embedding_model=emb,
        vector_store=vdb,
        max_steps=3,
        default_top_k=2,
    )

    response = agent.query("What does RFC 1035 specify?")
    assert isinstance(response, AgentResponse)
    assert "DNS is defined" in response.answer
    assert len(response.trajectory) >= 1
    assert response.total_steps == len(response.trajectory)
    assert response["answer"] == response.answer  # test __getitem__


# =====================================================================
# Tests: RAGService Facade
# =====================================================================

def test_rag_service_and_agentic_facade():
    emb = MockEmbeddingModel()
    vdb = InMemoryVectorStore()
    llm = MockLLMClient()

    service = RAGService(
        client=llm,
        embedding_service=emb,
        vector_db=vdb,
    )

    # Ingest text
    service.ingest("RFC 9000 specifies the QUIC transport protocol.", source="rfc9000.txt")
    assert vdb.count() == 1

    # Naive query
    naive_res = service.query("What does RFC 9000 define?", top_k=1)
    assert isinstance(naive_res, RAGResponse)
    assert naive_res.answer != ""

    # Upgrade to Agentic
    agentic_service = service.as_agentic(max_rounds=2)
    assert isinstance(agentic_service, AgenticRAGService)

    agentic_res = agentic_service.query("What protocol is in RFC 9000?")
    assert isinstance(agentic_res, RAGResponse)
    assert agentic_res.confidence > 0.0
