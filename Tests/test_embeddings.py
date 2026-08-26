import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.vecs.database.embeddings import docs_to_text, create, create_from_documents
from src.vecs.database.document import Document


def test_embeddings_sync_comprehensive():
    docs = [
        Document(text="First", metadata={"id": 1}),
        Document(text="Second", metadata={"id": 2}),
        Document(text="Third", metadata={"id": 3})
    ]
    texts = docs_to_text(docs)
    assert texts == ["First", "Second", "Third"]

    with patch('src.vecs.database.embeddings.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_response = Mock()
        mock_embed1 = Mock(embedding=[0.1, 0.2, 0.3])
        mock_embed2 = Mock(embedding=[0.4, 0.5, 0.6])
        mock_response.data = [mock_embed1, mock_embed2]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        result = create(["text1", "text2"], model="test-model", api_key="test-key")

        mock_openai.assert_called_once_with(api_key="test-key")
        mock_client.embeddings.create.assert_called_once_with(
            input=["text1", "text2"],
            model="test-model",
            encoding_format="float"
        )
        assert result.data == [mock_embed1, mock_embed2]

    with patch('src.vecs.database.embeddings.create') as mock_create:
        mock_response = Mock()
        mock_embed1 = Mock(embedding=[0.1, 0.2])
        mock_embed2 = Mock(embedding=[0.3, 0.4])
        mock_response.data = [mock_embed1, mock_embed2]
        mock_create.return_value = mock_response

        docs = [Document(text="Doc1"), Document(text="Doc2")]
        result = create_from_documents(docs, model="test-model")

        mock_create.assert_called_once()
        assert result.data == [mock_embed1, mock_embed2]


@patch('src.vecs.database.embeddings.AsyncOpenAI')
@pytest.mark.asyncio
async def test_embeddings_async_comprehensive(mock_async_openai):
    mock_client = Mock()
    mock_response = Mock()
    mock_embed1 = Mock(embedding=[0.1, 0.2, 0.3])
    mock_embed2 = Mock(embedding=[0.4, 0.5, 0.6])
    mock_response.data = [mock_embed1, mock_embed2]
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    mock_async_openai.return_value = mock_client

    from src.vecs.database.embeddings import acreate, acreate_from_documents

    result = await acreate(["text1", "text2"], model="test-model", api_key="test-key")
    mock_async_openai.assert_called_once_with(api_key="test-key")
    mock_client.embeddings.create.assert_called_once_with(
        input=["text1", "text2"],
        model="test-model",
        encoding_format="float"
    )
    assert result.data == [mock_embed1, mock_embed2]

    mock_client.embeddings.create.reset_mock()
    docs = [Document(text="AsyncDoc1"), Document(text="AsyncDoc2")]
    result = await acreate_from_documents(docs, model="test-model")
    mock_client.embeddings.create.assert_called_once_with(
        input=["AsyncDoc1", "AsyncDoc2"],
        model="test-model",
        encoding_format="float"
    )
    assert result.data == [mock_embed1, mock_embed2]
