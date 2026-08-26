import pytest
from src.vecs.hnsw.node import Node, distance


def test_node_comprehensive():
    assert distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert distance([1.0, 2.0], [4.0, 5.0]) == (1-4)**2 + (2-5)**2
    assert distance([5.0], [2.0]) == 9.0
    assert distance([-1.0, -2.0], [1.0, 2.0]) == (-1-1)**2 + (-2-2)**2

    node1 = Node(
        vector=[1.0, 2.0, 3.0],
        metadata={"id": 1, "tags": ["a", "b"]},
        content="test content",
        level=2
    )
    assert node1.vector == [1.0, 2.0, 3.0]
    assert node1.metadata == {"id": 1, "tags": ["a", "b"]}
    assert node1.content == "test content"
    assert len(node1.neighbors) == 3

    node2 = Node(vector=[4.0, 5.0], metadata={}, content="no level")
    assert node2.neighbors is None

    node1.neighbors[0].append(1)
    node1.neighbors[1].append(2)
    assert node1.neighbors[0] == [1]
    assert node1.neighbors[1] == [2]
