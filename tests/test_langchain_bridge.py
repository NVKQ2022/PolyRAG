"""Unit tests for LangChain document loader bridge and converter."""

from dataclasses import dataclass
from typing import Any
import pytest

from polyrag.app import PolyRAG
from polyrag.core.interfaces import BaseEmbeddingModel, BaseLLMClient
from polyrag.core.models import (
    Document as PolyDocument,
    LangChainDocumentConverter,
)
from polyrag.vector_stores.memory import InMemoryVectorStore


# Simulated LangChain Document class matching langchain_core.documents.Document
@dataclass
class SimulatedLangChainDocument:
    page_content: str
    metadata: dict[str, Any]
    id: str | None = None


class SimulatedLazyLoader:
    """Simulates a modern LangChain loader implementing lazy_load()."""

    def lazy_load(self):
        yield SimulatedLangChainDocument(
            page_content="Lazy Page 1: Introduction to OAuth",
            metadata={"source": "oauth.pdf", "page": 1},
            id="page-1",
        )
        yield SimulatedLangChainDocument(
            page_content="Lazy Page 2: PKCE Flow implementation",
            metadata={"source": "oauth.pdf", "page": 2},
            id="page-2",
        )


class SimulatedEagerLoader:
    """Simulates an eager LangChain loader implementing load()."""

    def load(self):
        return [
            SimulatedLangChainDocument(
                page_content="Eager Content from WebBaseLoader",
                metadata={"source": "https://example.com/api", "title": "API Docs"},
            )
        ]


class MockEmbedding(BaseEmbeddingModel):
    @property
    def dim(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class MockLLM(BaseLLMClient):
    @property
    def model_name(self) -> str:
        return "mock"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return "Answer based on context"

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "Chat reply"


# =========================================================================
# Tests
# =========================================================================

def test_converter_from_langchain_document():
    lc_doc = SimulatedLangChainDocument(
        page_content="Sample text content",
        metadata={"source": "report.pdf", "author": "Alice"},
        id="uuid-1234",
    )
    poly_doc = LangChainDocumentConverter.to_polyrag_document(lc_doc)

    assert isinstance(poly_doc, PolyDocument)
    assert poly_doc.text == "Sample text content"
    assert poly_doc.source == "report.pdf"
    assert poly_doc.metadata["author"] == "Alice"
    assert poly_doc.doc_id == "uuid-1234"


def test_converter_from_dict_and_polydoc():
    # From dict
    d = {"text": "dict text", "source": "data.json", "tag": "test"}
    res1 = LangChainDocumentConverter.to_polyrag_document(d)
    assert res1.text == "dict text"
    assert res1.source == "data.json"
    assert res1.metadata["tag"] == "test"

    # From PolyDocument
    pd = PolyDocument(source="original.txt", text="hello world", metadata={"a": 1})
    res2 = LangChainDocumentConverter.to_polyrag_document(pd)
    assert res2.text == "hello world"
    assert res2.source == "original.txt"


def test_ingest_documents_on_polyrag():
    vdb = InMemoryVectorStore()
    rag = PolyRAG.create(
        vector_store=vdb,
        embedding_model=MockEmbedding(),
        llm_client=MockLLM(),
    )

    lc_docs = [
        SimulatedLangChainDocument(
            page_content="Section 1: Distributed consensus algorithms",
            metadata={"source": "raft.pdf", "page": 1},
        ),
        SimulatedLangChainDocument(
            page_content="Section 2: Leader election dynamics",
            metadata={"source": "raft.pdf", "page": 2},
        ),
    ]

    indexed = rag.ingest_documents(lc_docs, metadata={"project": "consensus"})
    assert len(indexed) >= 2
    assert vdb.count() >= 2

    # Check that metadata merged correctly
    matches = vdb.search([1.0, 0.0], top_k=2)
    assert matches[0]["document"]["project"] == "consensus"
    assert matches[0]["document"]["source"] == "raft.pdf"


def test_ingest_langchain_lazy_loader():
    vdb = InMemoryVectorStore()
    rag = PolyRAG.create(
        vector_store=vdb,
        embedding_model=MockEmbedding(),
        llm_client=MockLLM(),
    )

    loader = SimulatedLazyLoader()
    indexed = rag.ingest_langchain_loader(loader)

    assert len(indexed) == 2
    assert vdb.count() == 2

    response = rag.query("What is OAuth?")
    assert response.answer == "Answer based on context"
    assert len(response.sources) >= 1


def test_ingest_langchain_eager_loader():
    vdb = InMemoryVectorStore()
    rag = PolyRAG.create(
        vector_store=vdb,
        embedding_model=MockEmbedding(),
        llm_client=MockLLM(),
    )

    loader = SimulatedEagerLoader()
    indexed = rag.ingest_langchain_loader(loader)

    assert len(indexed) == 1
    assert vdb.count() == 1


def test_invalid_loader_raises():
    vdb = InMemoryVectorStore()
    rag = PolyRAG.create(
        vector_store=vdb,
        embedding_model=MockEmbedding(),
        llm_client=MockLLM(),
    )

    with pytest.raises(TypeError, match="must implement"):
        rag.ingest_langchain_loader(object())
