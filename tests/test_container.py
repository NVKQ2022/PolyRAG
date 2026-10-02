"""Unit tests for PolyRAG Dependency Injection Container."""

import pytest

from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.container import Container
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.app import PolyRAG
from polyrag.exceptions import ConfigurationError
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent
from polyrag.vector_stores.memory import InMemoryVectorStore


class DummyEmbedding(BaseEmbeddingModel):
    @property
    def dim(self) -> int:
        return 2

    def embed_text(self, text: str) -> list[float]:
        return [0.5, 0.5]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]


class DummyLLM(BaseLLMClient):
    @property
    def model_name(self) -> str:
        return "dummy-model"

    def complete(self, prompt: str, **kwargs) -> str:
        return "Dummy answer"

    def complete_json(self, prompt: str, **kwargs) -> dict:
        return {"retrieval_needed": False, "answer": "Dummy JSON"}

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return "Dummy chat"


def test_container_register_instance_and_resolve():
    container = Container()
    emb = DummyEmbedding()
    container.register_instance(BaseEmbeddingModel, emb)

    assert container.is_registered(BaseEmbeddingModel)
    resolved = container.resolve(BaseEmbeddingModel)
    assert resolved is emb


def test_container_register_factory_singleton():
    container = Container()
    calls = 0

    def store_factory():
        nonlocal calls
        calls += 1
        return InMemoryVectorStore()

    container.register_factory(BaseVectorStore, store_factory, singleton=True)

    store1 = container.resolve(BaseVectorStore)
    store2 = container.resolve(BaseVectorStore)

    assert store1 is store2
    assert calls == 1


def test_container_register_factory_transient():
    container = Container()
    calls = 0

    def chunker_factory():
        nonlocal calls
        calls += 1
        return RecursiveCharacterChunker()

    container.register_factory(BaseChunker, chunker_factory, singleton=False)

    c1 = container.resolve(BaseChunker)
    c2 = container.resolve(BaseChunker)

    assert c1 is not c2
    assert calls == 2


def test_unregistered_interface_raises_error():
    container = Container()
    with pytest.raises(ConfigurationError):
        container.resolve(BaseLLMClient)


def test_container_pipeline_builders():
    container = Container.create(
        chunker=RecursiveCharacterChunker(),
        embedding_model=DummyEmbedding(),
        vector_store=InMemoryVectorStore(),
        llm_client=DummyLLM(),
    )

    # Naive RAG
    naive = container.build_naive_rag()
    assert isinstance(naive, NaiveRAG)
    assert isinstance(naive.embedding_model, DummyEmbedding)

    # Agentic RAG
    agentic = container.build_agentic_rag(top_k=2, max_rounds=1)
    assert isinstance(agentic, AgenticRAG)
    assert agentic.top_k == 2

    # ReAct Agent
    react = container.build_react_agent(max_steps=3)
    assert isinstance(react, ReActAgent)
    assert react.max_steps == 3

    # App & Pipeline Builders
    service = container.build_service()
    assert isinstance(service, PolyRAG)

    agentic_service = container.build_agentic_service(top_k=2)
    assert isinstance(agentic_service, AgenticRAG)


def test_rag_app_from_container():
    container = Container()
    container.register_instance(BaseChunker, RecursiveCharacterChunker())
    container.register_instance(BaseEmbeddingModel, DummyEmbedding())
    container.register_instance(BaseVectorStore, InMemoryVectorStore())
    container.register_instance(BaseLLMClient, DummyLLM())

    app = PolyRAG.from_container(container)
    assert isinstance(app, PolyRAG)

    # Ingest and query
    app.ingest("Sample text", source="doc.txt")
    results = app.retrieve("query", top_k=1)
    assert len(results) == 1
