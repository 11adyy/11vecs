"""Node representation and distance calculation for HNSW."""
import math

def distance(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)
    return 1 - cosine_similarity


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