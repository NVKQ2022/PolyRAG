"""Unit tests for MilvusVectorStore."""

from unittest.mock import MagicMock
import pytest

from polyrag.core.interfaces import BaseVectorStore
from polyrag.vector_stores.milvus import (
    MilvusLite,
    MilvusLiteVectorStore,
    MilvusVectorStore,
)



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


def test_milvus_missing_dependency_raises(monkeypatch):
    """Ensure meaningful ImportError is raised when pymilvus is absent."""
    import sys

    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(ImportError, match="pymilvus is required"):
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


def test_milvus_lite_instantiation_and_inheritance():
    """Ensure MilvusLiteVectorStore inherits from MilvusVectorStore and BaseVectorStore."""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    store = MilvusLiteVectorStore(
        db_path="./test_lite.db",
        collection_name="test_col",
        client=mock_client,
    )
    assert isinstance(store, MilvusVectorStore)
    assert isinstance(store, BaseVectorStore)
    assert isinstance(store, MilvusLiteVectorStore)
    assert MilvusLite is MilvusLiteVectorStore
    assert store.uri == "./test_lite.db"
    assert store.token == ""


def test_milvus_lite_db_path_parent_creation(tmp_path):
    """Ensure MilvusLiteVectorStore auto-creates parent directories if needed."""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    db_file = tmp_path / "deep" / "nested" / "vectors.db"
    assert not db_file.parent.exists()

    store = MilvusLiteVectorStore(
        db_path=db_file,
        collection_name="test_nested",
        client=mock_client,
    )
    assert db_file.parent.exists()
    assert store.uri == str(db_file)
    assert store.db_path == db_file


def test_milvus_lite_memory_path():
    """Ensure :memory: db_path works without creating directories."""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    store = MilvusLiteVectorStore(
        db_path=":memory:",
        client=mock_client,
    )
    assert store.uri == ":memory:"


def test_milvus_vector_store_lite_factory():
    """Ensure MilvusVectorStore.lite() factory method constructs MilvusLiteVectorStore."""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True

    store = MilvusVectorStore.lite(
        db_path="./factory_test.db",
        collection_name="factory_col",
        dimension=128,
        metric_type="L2",
        client=mock_client,
    )
    assert isinstance(store, MilvusLiteVectorStore)
    assert store.uri == "./factory_test.db"
    assert store.collection_name == "factory_col"
    assert store.dimension == 128
    assert store.metric_type == "L2"


def test_milvus_lite_module_exports():
    """Ensure MilvusLiteVectorStore is exported from all intended namespaces."""
    from polyrag import MilvusLite as PolyMilvusLite, MilvusLiteVectorStore as PolyMilvusLiteVectorStore
    from polyrag.vector_stores import (
        MilvusLite as VdbMilvusLite,
        MilvusLiteVectorStore as VdbMilvusLiteVectorStore,
    )
    from polyrag.vector_stores.milvus_lite import (
        MilvusLite as SubMilvusLite,
        MilvusLiteVectorStore as SubMilvusLiteVectorStore,
    )

    assert PolyMilvusLite is MilvusLiteVectorStore
    assert PolyMilvusLiteVectorStore is MilvusLiteVectorStore
    assert VdbMilvusLite is MilvusLiteVectorStore
    assert VdbMilvusLiteVectorStore is MilvusLiteVectorStore
    assert SubMilvusLite is MilvusLiteVectorStore
    assert SubMilvusLiteVectorStore is MilvusLiteVectorStore


def test_milvus_lite_operations(tmp_path):
    """Test full document workflow on MilvusLiteVectorStore."""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.search.return_value = [
        [
            {
                "id": "doc_1",
                "distance": 0.95,
                "entity": {"text": "Local embedded search result", "source": "local.txt"},
            }
        ]
    ]
    mock_client.query.return_value = [{"count(*)": 1}]

    db_path = tmp_path / "milvus_lite_test.db"
    store = MilvusLiteVectorStore(
        db_path=db_path,
        collection_name="embedded_docs",
        client=mock_client,
    )

    # 1. Add documents
    store.add_documents(
        vectors=[[0.1, 0.2]],
        documents=[{"_id": "doc_1", "text": "Local embedded search result", "source": "local.txt"}],
    )
    assert mock_client.insert.call_count == 1

    # 2. Count
    assert store.count() == 1

    # 3. Search
    results = store.search([0.1, 0.2], top_k=1)
    assert len(results) == 1
    assert results[0]["document"]["text"] == "Local embedded search result"
    assert results[0]["document"]["source"] == "local.txt"

    # 4. Clear
    store.clear()
    assert mock_client.drop_collection.call_count == 1

