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
        0: Node(vector=[10], metadata={}, content="far", level=1),
        1: Node(vector=[5], metadata={}, content="middle", level=1),
        2: Node(vector=[1], metadata={}, content="close", level=1),
    }

    hnsw._nodes[0].neighbors[1] = [1]
    hnsw._nodes[1].neighbors[1] = [2]

    hnsw._nodes[0].neighbors[0] = [1]
    hnsw._nodes[1].neighbors[0] = [2]

    hnsw._max_level = 1

    result = hnsw._greedy_search_until(
        query=[0],
        entry_point=0,
        start_level=1,
        finish_level=0,
    )

    assert result == 2


def test_greedy_search_includes_level_zero(hnsw):
    hnsw._nodes = {
        0: Node(vector=[10], metadata={}, content="far", level=1),
        1: Node(vector=[5], metadata={}, content="middle", level=1),
        2: Node(vector=[1], metadata={}, content="close", level=0),
    }

    hnsw._nodes[0].neighbors[1] = [1]
    hnsw._nodes[1].neighbors[1] = []

    hnsw._nodes[0].neighbors[0] = [1]
    hnsw._nodes[1].neighbors[0] = [2]

    hnsw._max_level = 1

    result = hnsw._greedy_search_until(
        query=[0],
        entry_point=0,
        start_level=1,
        finish_level=0,
    )

    assert result == 2


def test_ef_search_returns_nearest_nodes_and_handles_cycles(hnsw):
    hnsw._nodes = {
        0: Node(vector=[10], metadata={}, content="far", level=0),
        1: Node(vector=[6], metadata={}, content="middle", level=0),
        2: Node(vector=[4], metadata={}, content="near", level=0),
        3: Node(vector=[1], metadata={}, content="closest", level=0),
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
        ef_search=hnsw.ef_search,
    )

    assert result == [3, 2]
    assert len(result) == len(set(result))


def test_ef_search_returns_entry_point_when_it_has_no_neighbors(hnsw):
    hnsw._nodes = {
        0: Node(
            vector=[5],
            metadata={},
            content="only node",
            level=0,
        ),
    }

    hnsw._max_level = 0

    result = hnsw._ef_search(
        top_k=2,
        entry_point_id=0,
        query=[0],
        ef_search=hnsw.ef_search,
    )

    assert result == [0]


def test_search_returns_nearest_nodes(hnsw):
    hnsw._nodes = {
        0: Node(vector=[10], metadata={}, content="far", level=1),
        1: Node(vector=[6], metadata={}, content="middle", level=1),
        2: Node(vector=[3], metadata={}, content="near", level=0),
        3: Node(vector=[1], metadata={}, content="closest", level=0),
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

    assert [
        node.content
        for node in result
    ] == [
        "closest",
        "near",
    ]


def test_add_and_search(hnsw, monkeypatch):
    levels = iter([0, 0, 0, 0])

    monkeypatch.setattr(
        hnsw,
        "_random_level",
        lambda: next(levels),
    )

    nodes = [
        Node(
            vector=[10],
            metadata={},
            content="far",
        ),
        Node(
            vector=[7],
            metadata={},
            content="far-middle",
        ),
        Node(
            vector=[3],
            metadata={},
            content="near",
        ),
        Node(
            vector=[1],
            metadata={},
            content="closest",
        ),
    ]

    for node in nodes:
        hnsw.add(
            node.content,
            node.metadata,
            node.vector,
        )

    assert len(hnsw._nodes) == 4
    assert hnsw._entry_point is not None
    assert hnsw._max_level == 0

    for node in hnsw._nodes.values():
        assert len(node.neighbors) == 1

    assert any(
        node.neighbors[0]
        for node in hnsw._nodes.values()
    )

    result = hnsw.search(query=[0])

    assert [
        node.content
        for node in result
    ] == [
        "closest",
        "near",
    ]