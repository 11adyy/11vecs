from openai import OpenAI
from openai import AsyncOpenAI
from vecs.database.document import Document

def docs_to_text(documents: list[Document]) -> list[str]:
    return [doc.text for doc in documents]

async def acreate(texts : list[str], model: str, **kwargs):
    aclient = AsyncOpenAI(**kwargs)
    response = await aclient.embeddings.create(
        input=texts,
        model=model,
        encoding_format="float"
    )
    return response

def create(texts : list[str], model: str, **kwargs):
    client = OpenAI(**kwargs)
    response = client.embeddings.create(
        input=texts,
        model=model,
        encoding_format="float"
    )
    return response

async def acreate_from_documents(documents : list[Document], model: str, **kwargs):

    parsed_docs = docs_to_text(documents=documents)
    return await acreate(parsed_docs, model=model, **kwargs)

def create_from_documents(documents : list[Document], model: str, **kwargs):

    parsed_docs = docs_to_text(documents=documents)
    return create(texts=parsed_docs, model=model, **kwargs)
