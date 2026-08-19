from src.vectars.hnsw import HNSW, Node


def test_greedy_search_reaches_closest_node():
    hnsw = HNSW(
        top_k=2,
        ef_construction=10,
        ef_search=10,
        m=2,
    )

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

def test_greedy_search_includes_level_zero():
    hnsw = HNSW(
        top_k=2,
        ef_construction=10,
        ef_search=10,
        m=2,
    )

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
