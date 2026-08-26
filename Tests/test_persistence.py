import pytest
import json
import tempfile
from pathlib import Path
from src.vecs.database.persistence import save, load
from src.vecs.hnsw import HNSW, HnswState, Node


def test_persistence_save_load_structure():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = Path(f.name)

    try:
        hnsw = HNSW(top_k=5, ef_construction=200, ef_search=50, m=16)
        hnsw._entry_point = 0
        hnsw._max_level = 1

        node0 = Node(vector=[0.1, 0.2, 0.3], metadata={'id': 1}, content='test content', level=1)
        node0.neighbors = [[1], []]

        node1 = Node(vector=[0.4, 0.5, 0.6], metadata={'id': 2}, content='another content', level=1)
        node1.neighbors = [[0], []]

        hnsw._nodes = {0: node0, 1: node1}

        save(hnsw, temp_path)

        loaded_state = load(temp_path)

        assert loaded_state.entry_point == 0
        assert loaded_state.max_level == 1
        assert loaded_state.ef_construction == 200
        assert loaded_state.m == 16
        assert loaded_state.ef_search == 50
        assert loaded_state.top_k == 5
        assert len(loaded_state.nodes) == 2
        assert loaded_state.nodes[0].vector == [0.1, 0.2, 0.3]
        assert loaded_state.nodes[0].metadata == {'id': 1}
        assert loaded_state.nodes[0].content == 'test content'
        assert loaded_state.nodes[1].vector == [0.4, 0.5, 0.6]
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_persistence_state_validation():
    state = HnswState(
        entry_point=1,
        nodes={},
        max_level=2,
        ef_construction=100,
        m=8,
        ef_search=30,
        top_k=3
    )

    assert state.entry_point == 1
    assert state.max_level == 2
    assert state.ef_construction == 100
    assert state.m == 8
    assert state.ef_search == 30
    assert state.top_k == 3

    serialized = state.model_dump()
    assert 'entry_point' in serialized
    assert 'nodes' in serialized
    assert 'max_level' in serialized
    assert 'ef_construction' in serialized
    assert 'm' in serialized
    assert 'ef_search' in serialized
    assert 'top_k' in serialized


def test_persistence_load_from_invalid_file():
    nonexistent_path = Path("nonexistent_file.json")

    with pytest.raises(FileNotFoundError):
        load(nonexistent_path)
