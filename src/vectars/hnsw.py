def distance(a, b):
    return sum(
        (x - y) ** 2
        for x, y in zip(a, b)
    )

class Node:
    def __init__(self, *, vector, metadata, doc, level):
        self.vector = vector
        self.metadata = metadata
        self.doc = doc
        self.neighbors = [[] for _ in range(level + 1)]


class HNSW:
    def __init__(
        self,
        top_k: int,
        ef_construction: int,
        ef_search: int,
        m: int,
    ):
        self.top_k = top_k
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m

        self._entry_point: int | None = None
        self._nodes = {}
        self._max_level = -1

    def _greedy_search_once(
        self,
        query,
        entry_point: int,
        level: int,
    ) -> int:
        current = entry_point

        while True:
            current_distance = distance(
                query,
                self._nodes[current].vector,
            )
            best = current

            for neighbor_id in self._nodes[current].neighbors[level]:
                d = distance(
                    self._nodes[neighbor_id].vector,
                    query,
                )

                if d < current_distance:
                    current_distance = d
                    best = neighbor_id

            if best == current:
                return current

            current = best

    def _greedy_search_until_final(
        self,
        query,
        entry_point: int,
        level: int,
    ) -> int:
        if level < 0 or level > self._max_level:
            raise ValueError(
                f"level must be between 0 and {self._max_level}, got {level}"
            )

        for i_level in range(level, -1, -1):
            entry_point = self._greedy_search_once(
                query,
                entry_point,
                i_level,
            )

        return entry_point