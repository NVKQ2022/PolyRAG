"""Unit tests for chunker configuration and per-call parameter overrides."""

from pathlib import Path
import tempfile
from typing import Any
import pytest

from polyrag import (
    AdvancedRAG,
    AgenticRAG,
    Container,
    NaiveRAG,
    PolyRAG,
    ReActAgent,
)
from polyrag.chunkers import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    resolve_chunker,
)
from polyrag.core.interfaces import (
    BaseEmbeddingModel,
    BaseLLMClient,
)
from polyrag.core.models import Document
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
        return "Grounded synthesis response."

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return {"answer": "Mocked JSON response"}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "Mock chat response"


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
# Chunker Resolution Tests
# =============================================================================

def test_resolve_chunker_defaults():
    chunker = resolve_chunker(None)
    assert isinstance(chunker, RecursiveCharacterChunker)
    assert chunker.chunk_size == 550
    assert chunker.chunk_overlap == 35


def test_resolve_chunker_instance():
    custom = FixedSizeChunker(chunk_size=120, chunk_overlap=15)
    resolved = resolve_chunker(custom)
    assert resolved is custom
    assert resolved.chunk_size == 120


def test_resolve_chunker_string_aliases():
    fixed = resolve_chunker("fixed_size", chunk_size=200, chunk_overlap=20)
    assert isinstance(fixed, FixedSizeChunker)
    assert fixed.chunk_size == 200

    fixed_short = resolve_chunker("fixed")
    assert isinstance(fixed_short, FixedSizeChunker)

    rec = resolve_chunker("recursive", chunk_size=300, chunk_overlap=30)
    assert isinstance(rec, RecursiveCharacterChunker)
    assert rec.chunk_size == 300

    rec_char = resolve_chunker("recursive_character")
    assert isinstance(rec_char, RecursiveCharacterChunker)


def test_resolve_chunker_invalid():
    with pytest.raises(ValueError, match="Unknown chunker strategy"):
        resolve_chunker("unknown_strategy")

    with pytest.raises(TypeError, match="Expected BaseChunker instance or string strategy"):
        resolve_chunker(12345)  # type: ignore[arg-type]


# =============================================================================
# PolyRAG Instance Chunker Configuration Tests
# =============================================================================

def test_polyrag_init_with_string_chunker():
    rag = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
        chunker="fixed_size",
    )
    assert isinstance(rag.chunker, FixedSizeChunker)


def test_polyrag_init_with_instance_chunker():
    custom = FixedSizeChunker(chunk_size=150, chunk_overlap=10)
    rag = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
        chunker=custom,
    )
    assert rag.chunker is custom
    assert rag.chunker.chunk_size == 150


def test_polyrag_create_factory_with_chunker():
    rag = PolyRAG.create(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
        chunker="fixed",
    )
    assert isinstance(rag.chunker, FixedSizeChunker)


# =============================================================================
# Per-Call Ingestion Chunker Override Tests
# =============================================================================

def test_ingest_text_chunker_override():
    rag = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
        chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=0),
    )
    long_text = "A" * 200

    # Default chunker (500 chars) -> 1 chunk
    default_chunks = rag.ingest_text(long_text, source="default_run")
    assert len(default_chunks) == 1

    # Overridden with FixedSizeChunker(50) -> 4 chunks
    override_chunks = rag.ingest_text(
        long_text,
        source="override_run",
        chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=0),
    )
    assert len(override_chunks) == 4
    for c in override_chunks:
        assert len(c["text"]) <= 50

    # Instance chunker was not mutated
    assert rag.chunker.chunk_size == 500


def test_ingest_documents_chunker_override():
    rag = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    doc = Document(source="doc-1", text="B" * 150, metadata={"title": "Test Doc"})

    chunks = rag.ingest_documents(
        [doc],
        chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=0),
    )
    assert len(chunks) == 3


def test_ingest_file_chunker_override():
    rag = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write("C" * 180)
        temp_path = f.name

    try:
        chunks = rag.ingest_file(
            temp_path,
            chunker=FixedSizeChunker(chunk_size=60, chunk_overlap=0),
        )
        assert len(chunks) == 3
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_ingest_langchain_loader_chunker_override():
    rag = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    loader = MockLangChainLoader([
        MockLangChainDoc("D" * 120, {"author": "LangChain"}),
    ])

    chunks = rag.ingest_langchain_loader(
        loader,
        chunker=FixedSizeChunker(chunk_size=40, chunk_overlap=0),
    )
    assert len(chunks) == 3


def test_ingest_langchain_documents_chunker_override():
    rag = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
    )
    lc_docs = [
        MockLangChainDoc("E" * 100, {"topic": "AI"}),
    ]

    chunks = rag.ingest_langchain_documents(
        lc_docs,
        chunker="fixed",
    )
    assert len(chunks) > 0


# =============================================================================
# Pipeline Creation Chunker Override Tests
# =============================================================================

def test_pipeline_creation_chunker_overrides():
    rag = PolyRAG(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
        chunker="recursive",
    )

    naive = rag.create_naive_rag(chunker="fixed")
    assert isinstance(naive.chunker, FixedSizeChunker)

    advanced = rag.create_advanced_rag(chunker=FixedSizeChunker(75))
    assert isinstance(advanced.chunker, FixedSizeChunker)
    assert advanced.chunker.chunk_size == 75

    agentic = rag.create_agentic_rag(chunker="fixed_size")
    assert isinstance(agentic.chunker, FixedSizeChunker)

    react = rag.create_react_agent(chunker="fixed")
    assert isinstance(react.chunker, FixedSizeChunker)


# =============================================================================
# Container Chunker Support Tests
# =============================================================================

def test_container_create_with_chunker():
    container = Container.create(
        embedding_model=MockEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=MockLLM(),
        chunker="fixed_size",
    )
    naive = container.build_naive_rag()
    assert isinstance(naive.chunker, FixedSizeChunker)

    # Chunker override during build
    adv = container.build_advanced_rag(chunker=FixedSizeChunker(88))
    assert isinstance(adv.chunker, FixedSizeChunker)
    assert adv.chunker.chunk_size == 88

    agentic = container.build_agentic_rag(chunker="fixed")
    assert isinstance(agentic.chunker, FixedSizeChunker)

    react = container.build_react_agent(chunker="fixed")
    assert isinstance(react.chunker, FixedSizeChunker)
