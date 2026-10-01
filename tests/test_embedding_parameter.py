"""Unit tests for embedding configuration, resolution, adapters, and per-call overrides."""

from pathlib import Path
import tempfile
from typing import Any
import pytest

from polyrag import (
    AdvancedRAG,
    AgenticRAG,
    Container,
    LangChainEmbeddingAdapter,
    NaiveRAG,
    PolyRAG,
    ReActAgent,
    SentenceTransformerEmbedding,
    resolve_embedding_model,
)
from polyrag.core.interfaces import (
    BaseEmbeddingModel,
    BaseLLMClient,
)
from polyrag.core.models import Document
from polyrag.vector_stores.memory import InMemoryVectorStore


class MockEmbeddingA(BaseEmbeddingModel):
    def __init__(self, tag: str = "A"):
        self.tag = tag

    @property
    def dim(self) -> int:
        return 4

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class MockEmbeddingB(BaseEmbeddingModel):
    def __init__(self, tag: str = "B"):
        self.tag = tag

    @property
    def dim(self) -> int:
        return 4

    def embed_text(self, text: str) -> list[float]:
        return [0.9, 0.8, 0.7, 0.6]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [[0.9, 0.8, 0.7, 0.6] for _ in texts]


class MockLLM(BaseLLMClient):
    @property
    def model_name(self) -> str:
        return "mock-llm"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return "Grounded synthesis response."

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return {"answer": "Mocked JSON response"}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "Mock chat response"


class SimulatedLangChainEmbeddings:
    """Simulates a LangChain Embeddings class conforming to embed_documents / embed_query."""

    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * self.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * self.dimension for _ in texts]


class MockLangChainDoc:
    def __init__(self, page_content: str, metadata: dict[str, Any] | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockLangChainLoader:
    def __init__(self, docs: list[MockLangChainDoc]):
        self._docs = docs

    def load(self) -> list[MockLangChainDoc]:
        return self._docs


# =============================================================================
# Embedding Resolution Tests
# =============================================================================

def test_resolve_embedding_model_defaults():
    emb = resolve_embedding_model(None)
    assert isinstance(emb, SentenceTransformerEmbedding)
    assert emb.model_name == "all-MiniLM-L6-v2"


def test_resolve_embedding_model_instance():
    custom = MockEmbeddingA()
    resolved = resolve_embedding_model(custom)
    assert resolved is custom


def test_resolve_embedding_model_string_aliases():
    st = resolve_embedding_model("sentence_transformers")
    assert isinstance(st, SentenceTransformerEmbedding)

    st2 = resolve_embedding_model("local")
    assert isinstance(st2, SentenceTransformerEmbedding)

    # String model name
    hf = resolve_embedding_model("all-MiniLM-L6-v2")
    assert isinstance(hf, SentenceTransformerEmbedding)


def test_resolve_embedding_model_langchain_duck_typing():
    lc_emb = SimulatedLangChainEmbeddings(dimension=8)
    resolved = resolve_embedding_model(lc_emb)

    assert isinstance(resolved, LangChainEmbeddingAdapter)
    assert resolved.dim == 8

    # Verify query and batch embedding
    q_vec = resolved.embed_text("hello query")
    assert len(q_vec) == 8
    assert q_vec == [0.5] * 8

    b_vecs = resolved.embed_batch(["text1", "text2"])
    assert len(b_vecs) == 2
    assert len(b_vecs[0]) == 8


def test_resolve_embedding_model_invalid():
    with pytest.raises(TypeError, match="Expected BaseEmbeddingModel instance"):
        resolve_embedding_model(12345)


# =============================================================================
# PolyRAG Instance Embedding Configuration Tests
# =============================================================================

def test_polyrag_init_with_embedding_instance():
    custom = MockEmbeddingA()
    rag = PolyRAG(
        embedding_model=custom,
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    assert rag.embedding_model is custom


def test_polyrag_init_with_embedding_alias_kwarg():
    custom = MockEmbeddingB()
    rag = PolyRAG(
        embedding=custom,
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    assert rag.embedding_model is custom


def test_polyrag_init_with_langchain_embeddings():
    lc_emb = SimulatedLangChainEmbeddings(dimension=6)
    rag = PolyRAG(
        embedding=lc_emb,
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    assert isinstance(rag.embedding_model, LangChainEmbeddingAdapter)
    assert rag.embedding_model.dim == 6


def test_polyrag_create_factory_with_embedding():
    custom = MockEmbeddingA()
    rag = PolyRAG.create(
        embedding=custom,
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    assert rag.embedding_model is custom


# =============================================================================
# Per-Call Ingestion and Retrieval Embedding Override Tests
# =============================================================================

def test_ingest_text_embedding_override():
    base_emb = MockEmbeddingA()
    override_emb = MockEmbeddingB()

    vdb = InMemoryVectorStore()
    rag = PolyRAG(
        embedding_model=base_emb,
        vector_store=vdb,
        llm_client=MockLLM(),
    )

    # Ingest using override embedding model
    chunks = rag.ingest_text(
        "Sample text content for testing embedding override",
        source="doc_override",
        embedding_model=override_emb,
    )
    assert len(chunks) == 1
    # Check that vectors stored in vdb match MockEmbeddingB vectors ([0.9, 0.8, 0.7, 0.6])
    assert len(vdb.vectors) == 1
    assert vdb.vectors[0] == [0.9, 0.8, 0.7, 0.6]


def test_retrieve_embedding_override():
    base_emb = MockEmbeddingA()
    override_emb = MockEmbeddingB()

    vdb = InMemoryVectorStore()
    vdb.add_documents(
        vectors=[[0.9, 0.8, 0.7, 0.6]],
        documents=[{"source": "doc1", "text": "target content"}],
    )

    rag = PolyRAG(
        embedding_model=base_emb,
        vector_store=vdb,
        llm_client=MockLLM(),
    )

    # Retrieve using override embedding model
    results = rag.retrieve("query", embedding_model=override_emb)
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(1.0, rel=1e-3)


def test_ingest_documents_and_loader_embedding_override():
    override_emb = MockEmbeddingB()
    rag = PolyRAG(
        embedding_model=MockEmbeddingA(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )

    # Ingest document with override
    doc = Document(source="doc-b", text="Document text B")
    chunks = rag.ingest_documents([doc], embedding_model=override_emb)
    assert len(chunks) == 1

    # Ingest loader with override
    loader = MockLangChainLoader([MockLangChainDoc("Loader text")])
    chunks_loader = rag.ingest_langchain_loader(loader, embedding_model=override_emb)
    assert len(chunks_loader) == 1


# =============================================================================
# Pipeline Creation Embedding Override Tests
# =============================================================================

def test_pipeline_creation_embedding_overrides():
    rag = PolyRAG(
        embedding_model=MockEmbeddingA(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    override = MockEmbeddingB()

    naive = rag.create_naive_rag(embedding_model=override)
    assert naive.embedding_model is override

    advanced = rag.create_advanced_rag(embedding=override)
    assert advanced.embedding_model is override

    agentic = rag.create_agentic_rag(embedding_model=override)
    assert agentic.embedding_model is override

    react = rag.create_react_agent(embedding=override)
    assert react.embedding_model is override


# =============================================================================
# Container Embedding Support Tests
# =============================================================================

def test_container_create_with_embedding():
    custom = MockEmbeddingA()
    container = Container.create(
        embedding=custom,
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    assert container.resolve(BaseEmbeddingModel) is custom

    # Override when building pipeline
    override = MockEmbeddingB()
    naive = container.build_naive_rag(embedding_model=override)
    assert naive.embedding_model is override

    adv = container.build_advanced_rag(embedding=override)
    assert adv.embedding_model is override

    agentic = container.build_agentic_rag(embedding_model=override)
    assert agentic.embedding_model is override

    react = container.build_react_agent(embedding=override)
    assert react.embedding_model is override
