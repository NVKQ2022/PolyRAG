"""Tests for expanded parameters across PolyRAG vector stores."""

from typing import Any
import pytest

from polyrag.vector_stores import InMemoryVectorStore, resolve_vector_store


# =========================================================================
# InMemoryVectorStore Expanded Tests
# =========================================================================

def test_memory_initial_data():
    docs = [{"_id": "1", "text": "hello", "author": "Alice"}]
    vecs = [[1.0, 0.0]]
    store = InMemoryVectorStore(initial_documents=docs, initial_vectors=vecs)
    assert store.count() == 1
    assert store.peek(1)["documents"] == ["hello"]


def test_memory_where_filter():
    store = InMemoryVectorStore()
    docs = [
        {"_id": "1", "text": "DNS guide", "category": "Network"},
        {"_id": "2", "text": "Token guide", "category": "Auth"},
    ]
    vecs = [[1.0, 0.0], [0.9, 0.1]]
    store.add_documents(vecs, docs)

    # Filter for category="Auth"
    results = store.search([1.0, 0.0], top_k=5, where={"category": "Auth"})
    assert len(results) == 1
    assert results[0]["document"]["_id"] == "2"


def test_memory_filter_fn():
    store = InMemoryVectorStore()
    docs = [
        {"_id": "1", "text": "Old doc", "year": 2020},
        {"_id": "2", "text": "New doc", "year": 2024},
    ]
    vecs = [[1.0, 0.0], [1.0, 0.0]]
    store.add_documents(vecs, docs)

    results = store.search([1.0, 0.0], top_k=5, filter_fn=lambda d: d.get("year", 0) > 2022)
    assert len(results) == 1
    assert results[0]["document"]["year"] == 2024


def test_memory_l2_and_dot_metrics():
    # L2 metric
    store_l2 = InMemoryVectorStore(metric="l2")
    store_l2.add_documents([[1.0, 0.0]], [{"text": "a"}])
    res_l2 = store_l2.search([1.0, 0.0], top_k=1)
    assert res_l2[0]["distance"] == 0.0
    assert res_l2[0]["score"] == 1.0

    # Dot metric
    store_dot = InMemoryVectorStore(metric="dot")
    store_dot.add_documents([[2.0, 3.0]], [{"text": "b"}])
    res_dot = store_dot.search([1.0, 1.0], top_k=1)
    assert res_dot[0]["score"] == 5.0


def test_resolve_vector_store():
    # 1. Default to InMemoryVectorStore
    store = resolve_vector_store()
    assert isinstance(store, InMemoryVectorStore)

    # 2. String alias 'memory'
    store_mem = resolve_vector_store("memory")
    assert isinstance(store_mem, InMemoryVectorStore)

    # 3. Instance passthrough
    custom = InMemoryVectorStore(metric="l2")
    assert resolve_vector_store(custom) is custom

    # 4. Unknown string raises error
    with pytest.raises(TypeError, match="Expected VectorStore instance"):
        resolve_vector_store(12345)
