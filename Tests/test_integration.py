import pytest
import os
from pathlib import Path
from src.vecs.database.vector_database import VectorDatabase
from src.vecs.database.document import Document


@pytest.mark.integration
def test_vector_database_integration():
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-embed-1b:free")

    if not api_key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")

    vdb = VectorDatabase(
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_k=3,
        ef_construction=50,
        ef_search=20,
        m=8
    )

    documents = [
        Document(
            text="Python is a high-level programming language known for its simplicity.",
            metadata={"category": "programming", "language": "python"}
        ),
        Document(
            text="JavaScript is commonly used for web development and interactive websites.",
            metadata={"category": "programming", "language": "javascript"}
        ),
        Document(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "ai", "topic": "ml"}
        ),
        Document(
            text="Neural networks are computing systems inspired by biological neural networks.",
            metadata={"category": "ai", "topic": "deep learning"}
        ),
        Document(
            text="Natural language processing deals with the interaction between computers and human language.",
            metadata={"category": "ai", "topic": "nlp"}
        )
    ]

    vdb.add_documents(documents)

    assert len(vdb.hnsw._nodes) == 5
    assert vdb.hnsw._entry_point is not None

    results = vdb.search("artificial intelligence and machine learning")

    assert len(results) > 0
    assert len(results) <= 3

    result_contents = [node.content for node in results]
    assert any("artificial intelligence" in content.lower() or "machine learning" in content.lower() 
               for content in result_contents)

    results_programming = vdb.search("web development and programming languages")

    assert len(results_programming) > 0
    assert len(results_programming) <= 3

    result_programming_contents = [node.content for node in results_programming]
    assert any("javascript" in content.lower() or "programming" in content.lower() 
               for content in result_programming_contents)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_database_async_integration():
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-embed-1b:free")

    if not api_key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")

    vdb = await VectorDatabase.afrom_documents(
        documents=[
            Document(
                text="Async programming allows for non-blocking operations.",
                metadata={"category": "programming", "topic": "async"}
            ),
            Document(
                text="Database persistence ensures data survives program termination.",
                metadata={"category": "database", "topic": "persistence"}
            ),
            Document(
                text="Vector databases are optimized for similarity search.",
                metadata={"category": "database", "topic": "vectors"}
            )
        ],
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_k=2,
        ef_search=10,
        ef_construction=10,
        m=8
    )

    assert len(vdb.hnsw._nodes) == 3

    results = await vdb.asearch("database operations and vector search")

    assert len(results) > 0
    assert len(results) <= 2

    result_contents = [node.content for node in results]
    assert any("database" in content.lower() or "vector" in content.lower() 
               for content in result_contents)


@pytest.mark.integration
def test_vector_database_with_persistence():
    import tempfile

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-embed-1b:free")

    if not api_key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "hnsw_state.json"

        vdb1 = VectorDatabase.from_documents(
            documents=[
                Document(
                    text="First document about testing persistence.",
                    metadata={"id": 1}
                ),
                Document(
                    text="Second document about persistence testing.",
                    metadata={"id": 2}
                )
            ],
            model=model,
            base_url=base_url,
            api_key=api_key,
            top_k=2,
            ef_search=10,
            ef_construction=10,
            m=2,
            persistent_dir=None
        )

        assert len(vdb1.hnsw._nodes) == 2

        from src.vecs.database.persistence import save
        save(vdb1.hnsw, temp_path)

        vdb2 = VectorDatabase(
            model=model,
            base_url=base_url,
            api_key=api_key,
            top_k=2,
            ef_search=10,
            persistent_dir=temp_path
        )

        assert len(vdb2.hnsw._nodes) == 2
        assert vdb2.hnsw._entry_point == vdb1.hnsw._entry_point

        results = vdb2.search("persistence testing")

        assert len(results) > 0
        result_contents = [node.content for node in results]
        assert any("persistence" in content.lower() for content in result_contents)
