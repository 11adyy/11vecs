import pytest
from unittest.mock import patch
from src.vecs.hnsw import HNSW, Node
from src.vecs.hnsw.add import add, random_level


def test_add_comprehensive():
    hnsw = HNSW(top_k=5, ef_construction=10, ef_search=3, m=2)

    with patch('src.vecs.hnsw.add.random') as mock_random:
        mock_random.return_value = 0.5
        level = random_level(hnsw)
        assert isinstance(level, int)
        assert level >= 0

    hnsw.m = 1
    with pytest.raises(ValueError, match="m must be greater than 1"):
        random_level(hnsw)

    hnsw.m = 2

    with patch('src.vecs.hnsw.add.random_level') as mock_level:
        mock_level.return_value = 0
        node_id = add(hnsw, "first node", {"id": 1}, [1.0, 2.0])

    assert node_id == 0
    assert len(hnsw._nodes) == 1
    assert hnsw._entry_point == 0
    assert hnsw._nodes[0].content == "first node"
    assert hnsw._nodes[0].metadata == {"id": 1}
    assert hnsw._nodes[0].vector == [1.0, 2.0]

    with patch('src.vecs.hnsw.add.random_level') as mock_level:
        mock_level.return_value = 0
        for i in range(3):
            add(hnsw, f"node {i}", {"id": i}, [float(i), float(i)])

    assert len(hnsw._nodes) == 4

    hnsw._max_level = 0
    with patch('src.vecs.hnsw.add.random_level') as mock_level:
        mock_level.return_value = 2
        add(hnsw, "high level", {"id": 99}, [9.0, 9.0])

    assert hnsw._max_level == 2
