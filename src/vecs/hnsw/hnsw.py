"""Main HNSW index and public operations."""

from .add import add
from .search import search

from pydantic import BaseModel

class HnswState(BaseModel):
    entry_point: int | None = None
    nodes: dict
    max_level: int

    ef_construction: int
    m: int

    ef_search: int
    top_k: int

class HNSW:
    def __init__(
        self,
        top_k: int = 5,
        ef_construction: int  = 200,
        ef_search: int  = 50,
        m: int | None  = 16,
    ):
        self.top_k = top_k
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m

        self._entry_point = None
        self._nodes = {}
        self._max_level = -1

    @classmethod
    def from_persistent(cls, state: HnswState, top_k: int | None = None, ef_search: int | None = None) -> HNSW:
        top_k = top_k if top_k else state.top_k
        ef_search = ef_search if ef_search else state.ef_search

        instance = cls(top_k=top_k, ef_search=ef_search, ef_construction=state.ef_construction, m=state.m)

        instance._entry_point = state.entry_point
        instance._max_level = state.max_level
        instance._nodes = state.nodes
        return instance

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