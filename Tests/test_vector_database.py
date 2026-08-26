import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from src.vecs.database.vector_database import VectorDatabase
from src.vecs.database.document import Document


def test_vector_database_init_and_validation():
    with pytest.raises(ValueError, match="top_k parameter is required"):
        VectorDatabase(model="test", base_url="http://test.com", top_k=None)

    with pytest.raises(ValueError, match="Persistent directory cannot be used with"):
        VectorDatabase(
            model="test",
            base_url="http://test.com",
            top_k=5,
            persistent_dir=Path("test"),
            ef_construction=200
        )

    with patch('src.vecs.database.vector_database.HNSW') as mock_hnsw:
        mock_instance = Mock()
        mock_hnsw.return_value = mock_instance

        vdb = VectorDatabase(
            model="test-model",
            base_url="http://test.com",
            api_key="test-key",
            top_k=5,
            ef_construction=200,
            ef_search=50,
            m=16
        )

        assert vdb.model == "test-model"
        assert vdb.base_url == "http://test.com"
        assert vdb.api_key == "test-key"
        assert vdb.persistent_dir is None
        assert vdb.hnsw == mock_instance


def test_vector_database_operations_sync():
    with patch('src.vecs.database.vector_database.HNSW') as mock_hnsw, \
         patch('src.vecs.database.vector_database.create_from_documents') as mock_create, \
         patch('src.vecs.database.vector_database.create') as mock_search_create:

        mock_hnsw_instance = Mock()
        mock_hnsw.return_value = mock_hnsw_instance

        mock_embed_response = Mock()
        mock_embed_response.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
        mock_create.return_value = mock_embed_response

        mock_search_response = Mock()
        mock_search_response.data = [Mock(embedding=[0.5, 0.6])]
        mock_search_create.return_value = mock_search_response

        mock_hnsw_instance.search.return_value = ["result"]

        vdb = VectorDatabase(model="test", base_url="http://test.com", api_key="test-key", top_k=5)

        docs = [Document(text="Doc1"), Document(text="Doc2")]
        vdb.add_documents(docs)
        assert mock_hnsw_instance.add.call_count == 2

        result = vdb.search("query")
        mock_hnsw_instance.search.assert_called_once()
        assert result == ["result"]

        mock_hnsw_instance.reset_mock()
        VectorDatabase.from_documents(docs, model="test", base_url="http://test.com", api_key="test-key")
        assert mock_hnsw_instance.add.call_count == 2


@patch('src.vecs.database.vector_database.HNSW')
@pytest.mark.asyncio
async def test_vector_database_operations_async(mock_hnsw):
    mock_hnsw_instance = Mock()
    mock_hnsw.return_value = mock_hnsw_instance

    with patch('src.vecs.database.vector_database.acreate_from_documents') as mock_acreate, \
         patch('src.vecs.database.vector_database.acreate') as mock_asearch:

        mock_embed_response = Mock()
        mock_embed_response.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
        mock_acreate.return_value = mock_embed_response

        mock_search_response = Mock()
        mock_search_response.data = [Mock(embedding=[0.5, 0.6])]
        mock_asearch.return_value = mock_search_response

        mock_hnsw_instance.search.return_value = ["result"]

        vdb = VectorDatabase(model="test", base_url="http://test.com", api_key="test-key", top_k=5)

        docs = [Document(text="Doc1"), Document(text="Doc2")]
        await vdb.aadd_documents(docs)
        assert mock_hnsw_instance.add.call_count == 2

        result = await vdb.asearch("query")
        mock_hnsw_instance.search.assert_called_once()
        assert result == ["result"]

        mock_hnsw_instance.reset_mock()
        await VectorDatabase.afrom_documents(docs, model="test", base_url="http://test.com", api_key="test-key")
        assert mock_hnsw_instance.add.call_count == 2
