import pytest

from src.vectars.hnsw import HNSW, Node


@pytest.fixture
def hnsw():
    return HNSW(
        top_k=2,
        ef_construction=10,
        ef_search=3,
        m=2,
    )


def test_greedy_search_reaches_closest_node(hnsw):
    hnsw._nodes = {
        0: Node(vector=[10], metadata={}, doc="far", level=1),
        1: Node(vector=[5], metadata={}, doc="middle", level=1),
        2: Node(vector=[1], metadata={}, doc="close", level=1),
    }

    hnsw._nodes[0].neighbors[1] = [1]
    hnsw._nodes[1].neighbors[1] = [2]

    hnsw._nodes[0].neighbors[0] = [1]
    hnsw._nodes[1].neighbors[0] = [2]

    hnsw._max_level = 1

    result = hnsw._greedy_search_until_final(
        query=[0],
        entry_point=0,
        level=1,
    )

    assert result == 2


def test_greedy_search_includes_level_zero(hnsw):
    hnsw._nodes = {
        0: Node(vector=[10], metadata={}, doc="far", level=1),
        1: Node(vector=[5], metadata={}, doc="middle", level=1),
        2: Node(vector=[1], metadata={}, doc="close", level=0),
    }

    hnsw._nodes[0].neighbors[1] = [1]
    hnsw._nodes[1].neighbors[1] = []

    hnsw._nodes[0].neighbors[0] = [1]
    hnsw._nodes[1].neighbors[0] = [2]

    hnsw._max_level = 1

    result = hnsw._greedy_search_until_final(
        query=[0],
        entry_point=0,
        level=1,
    )

    assert result == 2


def test_ef_search_returns_nearest_nodes_and_handles_cycles(hnsw):
    hnsw._nodes = {
        0: Node(vector=[10], metadata={}, doc="far", level=0),
        1: Node(vector=[6], metadata={}, doc="middle", level=0),
        2: Node(vector=[4], metadata={}, doc="near", level=0),
        3: Node(vector=[1], metadata={}, doc="closest", level=0),
    }

    hnsw._nodes[0].neighbors[0] = [1, 2]
    hnsw._nodes[1].neighbors[0] = [0, 3]
    hnsw._nodes[2].neighbors[0] = [0, 1]
    hnsw._nodes[3].neighbors[0] = [1]

    hnsw._max_level = 0

    result = hnsw._ef_search(
        top_k=2,
        entry_point_id=0,
        query=[0],
        ef_search=hnsw.ef_search
    )

    assert result == [3, 2]
    assert len(result) == len(set(result))


def test_ef_search_returns_entry_point_when_it_has_no_neighbors(hnsw):
    hnsw._nodes = {
        0: Node(vector=[5], metadata={}, doc="only node", level=0),
    }
    hnsw._max_level = 0

    result = hnsw._ef_search(
        top_k=2,
        entry_point_id=0,
        query=[0],
        ef_search=hnsw.ef_search
    )
    assert result == [0]

def test_search_returns_nearest_nodes(hnsw):
    hnsw._nodes = {
        0: Node(vector=[10], metadata={}, doc="far", level=1),
        1: Node(vector=[6], metadata={}, doc="middle", level=1),
        2: Node(vector=[3], metadata={}, doc="near", level=0),
        3: Node(vector=[1], metadata={}, doc="closest", level=0),
    }

    hnsw._nodes[0].neighbors[1] = [1]
    hnsw._nodes[1].neighbors[1] = []

    hnsw._nodes[0].neighbors[0] = [1]
    hnsw._nodes[1].neighbors[0] = [2]
    hnsw._nodes[2].neighbors[0] = [3]
    hnsw._nodes[3].neighbors[0] = [2]

    hnsw._entry_point = 0
    hnsw._max_level = 1

    result = hnsw.search(query=[0])

    assert [node.doc for node in result] == ["closest", "near"]