"""Main HNSW index and public operations."""

from .add import add
from .search import search


class HNSW:
    def __init__(
        self,
        top_k,
        ef_construction,
        ef_search,
        m,
    ):
        self.top_k = top_k
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m

        self._entry_point = None
        self._nodes = {}
        self._max_level = -1

    def search(self, query):
        return search(
            self,
            query,
        )

    def add(
        self,
        content,
        metadata,
        vector,
    ):
        return add(
            self,
            content,
            metadata,
            vector,
        )

    def delete(self, node_id):
        pass

    def edit(self, node_id):
        pass