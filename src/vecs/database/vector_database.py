from pathlib import Path
from vecs.hnsw import HNSW
from .persistence import load as load_hnsw_file
from .embeddings import create_from_documents, acreate_from_documents, create, acreate


class VectorDatabase:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        persistent_dir: Path | None = None,

        top_k: int | None = None,
        ef_search: int | None = None,

        m: int | None = None,
        ef_construction: int | None = None,
    ):

        if persistent_dir and (
            ef_construction is not None or m is not None
        ):
            raise ValueError(
                "Persistent directory cannot be used with "
                "ef_construction or m parameters."
            )

        if top_k is None:
            raise ValueError("top_k parameter is required.")

        if persistent_dir:
            self.persistent_dir = persistent_dir

            state = load_hnsw_file(persistent_dir)

            self.hnsw = HNSW.from_persistent(
                state=state,
                top_k=top_k,
                ef_search=ef_search,
            )

        else:
            self.persistent_dir = None

            self.hnsw = HNSW(
                top_k,
                ef_construction,
                ef_search,
                m,
            )

        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    @classmethod
    def from_documents(
        cls,
        documents: list,
        model: str,
        base_url: str,
        api_key: str | None = None,
        top_k: int = 5,
        ef_search: int | None = None,
        m: int | None = None,
        ef_construction: int | None = None,
        persistent_dir: Path | None = None,
    ):
        vdb = cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            top_k=top_k,
            ef_search=ef_search,
            m=m,
            ef_construction=ef_construction,
            persistent_dir=persistent_dir,
        )

        vdb.add_documents(documents)

        return vdb

    @classmethod
    async def afrom_documents(
        cls,
        documents: list,
        model: str,
        base_url: str,
        api_key: str | None = None,
        top_k: int = 5,
        ef_search: int | None = None,
        m: int | None = None,
        ef_construction: int | None = None,
        persistent_dir: Path | None = None,
    ):
        vdb = cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            top_k=top_k,
            ef_search=ef_search,
            m=m,
            ef_construction=ef_construction,
            persistent_dir=persistent_dir,
        )

        await vdb.aadd_documents(documents)

        return vdb

    def add_documents(self, documents: list):
        embed_response = create_from_documents(
            documents=documents,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        for i, doc in enumerate(documents):
            self.hnsw.add(
                content=doc.text,
                metadata=doc.metadata,
                vector=embed_response.data[i].embedding,
            )

    async def aadd_documents(self, documents: list):
        embed_response = await acreate_from_documents(
            documents=documents,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        for i, doc in enumerate(documents):
            self.hnsw.add(
                content=doc.text,
                metadata=doc.metadata,
                vector=embed_response.data[i].embedding,
            )

    def search(self, query: str, **kwargs):
        embedded_query = create([query], model=self.model, base_url=self.base_url, api_key=self.api_key, **kwargs)
        return self.hnsw.search(query=embedded_query.data[0].embedding)

    async def asearch(self, query: str, **kwargs):
        embedded_query = await acreate([query], model=self.model, base_url=self.base_url, api_key=self.api_key, **kwargs)
        return self.hnsw.search(query=embedded_query.data[0].embedding)