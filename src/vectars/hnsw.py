import heapq
from itertools import count

def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

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

    def _get_best_edge(
            self,
            entry_node_id: int,
            query: list,
            level: int,
            blacklist: set | None = None,
    ) -> tuple[int | None, float | None]:

        if level < 0 or level > self._max_level:
            raise ValueError(
                f"level must be between 0 and {self._max_level}, got {level}"
            )

        best_id = None
        best_distance = None

        for neighbor_id in self._nodes[entry_node_id].neighbors[level]:
            if blacklist and neighbor_id in blacklist:
                continue

            d = distance(
                self._nodes[neighbor_id].vector,
                query,
            )

            if best_distance is None or d < best_distance:
                best_id = neighbor_id
                best_distance = d

        return best_id, best_distance

    def _ef_search(
            self,
            top_k: int,
            ef_search: int,
            entry_point_id: int,
            query: list,
            *,
            level: int = 0,
    ):
        candidates = []
        results = []
        counter = count()
        visited = {entry_point_id}

        entry_distance = distance(
            self._nodes[entry_point_id].vector,
            query,
        )

        heapq.heappush(
            candidates,
            (entry_distance, next(counter), entry_point_id),
        )
        heapq.heappush(
            results,
            (-entry_distance, next(counter), entry_point_id),
        )

        while candidates:
            current_distance, _, current_id = heapq.heappop(candidates)

            worst_result_distance = -results[0][0]

            if (
                    len(results) >= ef_search
                    and current_distance > worst_result_distance
            ):
                break

            for neighbor_id in self._nodes[current_id].neighbors[level]:
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)

                neighbor_distance = distance(
                    self._nodes[neighbor_id].vector,
                    query,
                )

                if (
                        len(results) < ef_search
                        or neighbor_distance < -results[0][0]
                ):
                    heapq.heappush(
                        candidates,
                        (neighbor_distance, next(counter), neighbor_id),
                    )
                    heapq.heappush(
                        results,
                        (-neighbor_distance, next(counter), neighbor_id),
                    )

                    if len(results) > ef_search:
                        heapq.heappop(results)

        ordered = sorted(results, key=lambda item: -item[0])
        return [node_id for _, _, node_id in ordered[:top_k]]

    def search(self, query) -> list[Node]:
        if self._entry_point is None:
            return []
        greedy_search_result = self._greedy_search_until_final(query, entry_point=self._entry_point, level=self._max_level)
        best_nodes = self._ef_search(entry_point_id=greedy_search_result, query=query, top_k=self.top_k, ef_search=self.ef_search, level=0)
        return [self._nodes[node_id] for node_id in best_nodes]