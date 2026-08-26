from pydantic import BaseModel

class Document(BaseModel):
    metadata: dict = {}
    text: str = ""
