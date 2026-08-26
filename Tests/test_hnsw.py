import pytest
from src.vecs.hnsw import HNSW, HnswState


def test_hnsw_initialization_and_state():
    hnsw1 = HNSW()
    assert hnsw1.top_k == 5
    assert hnsw1.ef_construction == 200
    assert hnsw1.ef_search == 50
    assert hnsw1.m == 16
    assert hnsw1._entry_point is None
    assert hnsw1._nodes == {}
    assert hnsw1._max_level == -1

    hnsw2 = HNSW(top_k=10, ef_construction=100, ef_search=20, m=8)
    assert hnsw2.top_k == 10
    assert hnsw2.ef_construction == 100
    assert hnsw2.ef_search == 20
    assert hnsw2.m == 8

    state = HnswState(
        entry_point=0,
        nodes={},
        max_level=2,
        ef_construction=200,
        m=16,
        ef_search=50,
        top_k=5
    )
    assert state.entry_point == 0
    assert state.max_level == 2


def test_hnsw_from_persistent():
    state = HnswState(
        entry_point=1,
        nodes={},
        max_level=1,
        ef_construction=150,
        m=12,
        ef_search=40,
        top_k=6
    )

    hnsw1 = HNSW.from_persistent(state)
    assert hnsw1.top_k == 6
    assert hnsw1.ef_search == 40
    assert hnsw1.ef_construction == 150
    assert hnsw1.m == 12
    assert hnsw1._entry_point == 1
    assert hnsw1._max_level == 1

    hnsw2 = HNSW.from_persistent(state, top_k=10, ef_search=20)
    assert hnsw2.top_k == 10
    assert hnsw2.ef_search == 20
    assert hnsw2.ef_construction == 150
    assert hnsw2.m == 12


def test_hnsw_basic_operations():
    hnsw = HNSW(top_k=3, ef_construction=10, ef_search=5, m=2)

    result = hnsw.add("test content", {"key": "value"}, [1.0, 2.0, 3.0])
    assert result == 0
    assert len(hnsw._nodes) == 1
    assert hnsw._entry_point == 0

    search_result = hnsw.search(query=[1.0, 2.0, 3.0])
    assert len(search_result) == 1
    assert search_result[0].content == "test content"

    empty_hnsw = HNSW()
    empty_result = empty_hnsw.search(query=[1.0, 2.0])
    assert empty_result == []

    assert hnsw.delete(0) is None
    assert hnsw.edit(0) is None
