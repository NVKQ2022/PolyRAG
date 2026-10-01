"""Unit tests for MilvusVectorStore."""

from unittest.mock import MagicMock
import pytest

from polyrag.core.interfaces import BaseVectorStore
from polyrag.vector_stores.milvus import MilvusVectorStore


def test_milvus_interface_compliance():
    """Ensure MilvusVectorStore implements BaseVectorStore interface."""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    store = MilvusVectorStore(
        uri="http://localhost:19530",
        collection_name="test_col",
        client=mock_client,
    )
    assert isinstance(store, BaseVectorStore)


def test_milvus_missing_dependency_raises():
    """Ensure meaningful ImportError is raised when pymilvus is absent."""
    with pytest.raises(ImportError, match="pymilvus is required"):
        # Without passing client, it will attempt `import pymilvus` which is not installed in test venv
        MilvusVectorStore(uri="http://localhost:19530")


def test_milvus_ensure_collection_on_init():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = False

    store = MilvusVectorStore(
        uri="http://localhost:19530",
        collection_name="test_col",
        dimension=128,
        metric_type="COSINE",
        client=mock_client,
    )

    mock_client.create_collection.assert_called_once_with(
        collection_name="test_col",
        dimension=128,
        primary_field_name="id",
        id_type="string",
        max_length=512,
        auto_id=False,
        metric_type="COSINE",
        enable_dynamic_field=True,
        consistency_level="Strong",
    )


def test_milvus_clear():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    store = MilvusVectorStore(
        collection_name="test_col",
        dimension=64,
        client=mock_client,
    )

    store.clear()
    mock_client.drop_collection.assert_called_with(collection_name="test_col")


def test_milvus_add_documents_validation():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    store = MilvusVectorStore(collection_name="test_col", client=mock_client)

    # Empty documents - no-op
    store.add_documents([], [])
    assert mock_client.insert.call_count == 0

    # Length mismatch
    with pytest.raises(ValueError, match="same length"):
        store.add_documents([[0.1, 0.2]], [])

    # Invalid batch_size
    with pytest.raises(ValueError, match="greater than 0"):
        store.add_documents([[0.1]], [{"text": "hi"}], batch_size=0)


def test_milvus_add_documents_success():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    store = MilvusVectorStore(collection_name="test_col", client=mock_client)

    vectors = [[0.1, 0.2], [0.3, 0.4]]
    documents = [
        {"_id": "doc_1", "text": "First doc", "source": "a.txt"},
        {"_id": "doc_2", "text": "Second doc", "source": "b.txt", "author": "Alice"},
    ]

    store.add_documents(vectors, documents, batch_size=10)

    assert mock_client.insert.call_count == 1
    call_args = mock_client.insert.call_args[1]
    assert call_args["collection_name"] == "test_col"
    inserted_data = call_args["data"]

    assert len(inserted_data) == 2
    assert inserted_data[0]["id"] == "doc_1"
    assert inserted_data[0]["text"] == "First doc"
    assert inserted_data[0]["vector"] == [0.1, 0.2]
    assert inserted_data[0]["source"] == "a.txt"

    assert inserted_data[1]["id"] == "doc_2"
    assert inserted_data[1]["author"] == "Alice"


def test_milvus_search():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    mock_hit_1 = {
        "id": "doc_1",
        "distance": 0.85,
        "entity": {
            "text": "First chunk",
            "metadata": {"source": "rfc1035.txt", "chunk_id": 1},
            "custom_tag": "dns",
        },
    }
    mock_client.search.return_value = [[mock_hit_1]]

    store = MilvusVectorStore(
        collection_name="test_col",
        metric_type="COSINE",
        client=mock_client,
    )

    results = store.search([0.1, 0.2], top_k=2)

    assert len(results) == 1
    assert results[0]["score"] == 0.85
    assert results[0]["distance"] == pytest.approx(0.15)
    doc = results[0]["document"]
    assert doc["_id"] == "doc_1"
    assert doc["text"] == "First chunk"
    assert doc["source"] == "rfc1035.txt"
    assert doc["custom_tag"] == "dns"


def test_milvus_search_l2():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    mock_hit = {
        "id": "doc_2",
        "distance": 1.0,
        "entity": {"text": "Hello"},
    }
    mock_client.search.return_value = [[mock_hit]]

    store = MilvusVectorStore(
        collection_name="test_col",
        metric_type="L2",
        client=mock_client,
    )

    results = store.search([0.1, 0.2], top_k=1)
    assert len(results) == 1
    assert results[0]["distance"] == 1.0
    assert results[0]["score"] == 0.5  # 1.0 / (1.0 + 1.0)


def test_milvus_count():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.query.return_value = [{"count(*)": 42}]

    store = MilvusVectorStore(collection_name="test_col", client=mock_client)
    assert store.count() == 42


def test_milvus_peek():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.query.return_value = [
        {"id": "doc_1", "text": "chunk 1", "metadata": {"source": "a.txt"}},
        {"id": "doc_2", "text": "chunk 2", "metadata": {"source": "b.txt"}},
    ]

    store = MilvusVectorStore(collection_name="test_col", client=mock_client)
    peeked = store.peek(limit=2)

    assert peeked["documents"] == ["chunk 1", "chunk 2"]
    assert peeked["ids"] == ["doc_1", "doc_2"]
