import pytest
from src.vecs.hnsw import HNSW, Node
from src.vecs.hnsw.search import greedy, greedy_range, beam, beam_range, search


def test_search_comprehensive():
    hnsw = HNSW(top_k=2, ef_construction=10, ef_search=3, m=2)

    hnsw._nodes = {
        0: Node(vector=[1.0, 0.0], metadata={}, content="target", level=1),
        1: Node(vector=[0.9, 0.1], metadata={}, content="similar", level=1),
        2: Node(vector=[0.0, 1.0], metadata={}, content="orthogonal", level=1),
    }
    hnsw._nodes[0].neighbors[1] = [1]
    hnsw._nodes[1].neighbors[1] = [2]
    hnsw._nodes[2].neighbors[1] = []
    hnsw._max_level = 1

    result = greedy(hnsw, query=[1.0, 0.0], entry_point_id=0, level=1)
    assert result in [0, 1, 2]

    result = greedy_range(hnsw, query=[1.0, 0.0], entry_point_id=0, start_level=1, end_level=0)
    assert result in [0, 1, 2]

    with pytest.raises(ValueError, match="level must be between"):
        greedy_range(hnsw, query=[1.0, 0.0], entry_point_id=0, start_level=5, end_level=0)

    hnsw._nodes = {
        0: Node(vector=[1.0, 0.0], metadata={}, content="target", level=0),
        1: Node(vector=[0.9, 0.1], metadata={}, content="similar", level=0),
        2: Node(vector=[0.0, 1.0], metadata={}, content="orthogonal", level=0),
        3: Node(vector=[-1.0, 0.0], metadata={}, content="opposite", level=0),
    }
    hnsw._nodes[0].neighbors[0] = [1, 2]
    hnsw._nodes[1].neighbors[0] = [0, 3]
    hnsw._nodes[2].neighbors[0] = [0, 1]
    hnsw._nodes[3].neighbors[0] = [1]
    hnsw._max_level = 0

    result = beam(hnsw, query=[1.0, 0.0], entry_point_id=0, level=0, beam_width=3, top_k=2)
    assert len(result) == 2
    assert all(r in [0, 1, 2, 3] for r in result)

    result = beam_range(hnsw, query=[1.0, 0.0], entry_point_id=0, start_level=0, end_level=0, beam_width=3, top_k=2)
    assert len(result) == 1

    with pytest.raises(ValueError, match="start_level cannot be negative"):
        beam_range(hnsw, query=[1.0, 0.0], entry_point_id=0, start_level=-1, end_level=0, beam_width=3, top_k=2)

    hnsw._entry_point = 0
    result = search(hnsw, query=[1.0, 0.0])
    assert len(result) == 2
    assert all(node.vector is not None for node in result)

    empty_hnsw = HNSW()
    empty_result = search(empty_hnsw, query=[1.0, 0.0])
    assert empty_result == []
