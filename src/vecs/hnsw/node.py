"""Node representation and distance calculation for HNSW."""

def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


class Node:
    def __init__(
        self,
        vector,
        metadata,
        content,
        level=None,
    ):
        self.vector = vector
        self.metadata = metadata
        self.content = content

        self.neighbors = (
            [[] for _ in range(level + 1)]
            if level is not None
            else None
        )