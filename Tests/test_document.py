import pytest
from src.vecs.database.document import Document


def test_document_comprehensive():
    doc1 = Document()
    assert doc1.text == ""
    assert doc1.metadata == {}

    doc2 = Document(
        text="Test content",
        metadata={"key": "value", "number": 42, "tags": ["a", "b"]}
    )
    assert doc2.text == "Test content"
    assert doc2.metadata == {"key": "value", "number": 42, "tags": ["a", "b"]}

    doc2.text = "Modified"
    doc2.metadata["new_key"] = "new_value"
    assert doc2.text == "Modified"
    assert doc2.metadata["new_key"] == "new_value"

    serialized = doc2.model_dump()
    assert serialized["text"] == "Modified"
    assert serialized["metadata"]["new_key"] == "new_value"
