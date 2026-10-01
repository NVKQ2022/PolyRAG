"""Tests for expanded parameters across all PolyRAG vector stores."""

from unittest.mock import MagicMock, patch
import pytest

from polyrag.vector_stores.chroma import ChromaVectorStore
from polyrag.vector_stores.memory import InMemoryVectorStore
from polyrag.vector_stores.milvus import MilvusVectorStore


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


# =========================================================================
# ChromaVectorStore Expanded Tests
# =========================================================================

def test_chroma_where_and_where_document():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_collection.query.return_value = {
        "ids": [["doc_1"]],
        "documents": [["DNS spec"]],
        "metadatas": [[{"source": "rfc1035.txt"}]],
        "distances": [[0.1]],
    }

    store = ChromaVectorStore(
        collection_name="test_col",
        client=mock_client,
    )

    results = store.search(
        [0.1, 0.2],
        top_k=3,
        where={"source": "rfc1035.txt"},
        where_document={"$contains": "DNS"},
    )

    assert len(results) == 1
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args[1]
    assert call_kwargs["where"] == {"source": "rfc1035.txt"}
    assert call_kwargs["where_document"] == {"$contains": "DNS"}


def test_chroma_http_client_initialization():
    mock_chromadb = MagicMock()
    with patch.dict("sys.modules", {"chromadb": mock_chromadb, "chromadb.config": MagicMock()}):
        ChromaVectorStore(
            host="192.168.1.100",
            port=9000,
            ssl=True,
            headers={"Authorization": "Bearer token"},
        )
        mock_chromadb.HttpClient.assert_called_once_with(
            host="192.168.1.100",
            port=9000,
            ssl=True,
            headers={"Authorization": "Bearer token"},
            settings=mock_chromadb.HttpClient.call_args[1]["settings"],
        )


# =========================================================================
# MilvusVectorStore Expanded Tests
# =========================================================================

def test_milvus_filter_expr_and_partition():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.search.return_value = [[
        {"id": "1", "distance": 0.9, "entity": {"text": "filtered doc", "dept": "Security"}}
    ]]

    store = MilvusVectorStore(
        collection_name="test_col",
        partition_name="tenant_alpha",
        client=mock_client,
    )

    # Insert with partition
    store.add_documents([[0.1, 0.2]], [{"text": "test"}])
    insert_call = mock_client.insert.call_args[1]
    assert insert_call["partition_name"] == "tenant_alpha"

    # Search with filter_expr
    results = store.search(
        [0.1, 0.2],
        top_k=2,
        filter_expr='dept == "Security"',
    )

    assert len(results) == 1
    search_call = mock_client.search.call_args[1]
    assert search_call["filter"] == 'dept == "Security"'
    assert search_call["partition_names"] == ["tenant_alpha"]


def test_milvus_client_kwargs_forwarding():
    mock_pymilvus = MagicMock()
    with patch.dict("sys.modules", {"pymilvus": mock_pymilvus}):
        MilvusVectorStore(
            uri="http://milvus:19530",
            token="secret",
            db_name="tenant_db",
            timeout=15.0,
            custom_arg="value",
        )
        mock_pymilvus.MilvusClient.assert_called_once_with(
            uri="http://milvus:19530",
            token="secret",
            db_name="tenant_db",
            timeout=15.0,
            custom_arg="value",
        )
