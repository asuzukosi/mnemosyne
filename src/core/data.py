from pydantic import BaseModel, Field


class DataItem(BaseModel):
    id: str
    key: str
    content: str
    metadata: dict = Field(default_factory=dict)

class QueryData(BaseModel):
    query: str
    context: str

class SearchResult(BaseModel):
    query: str
    context: str
    results: list[DataItem]


class SearchConfig(BaseModel):
    name: str
    description: str
    type: str
    config: dict

class DataLoaderConfig(BaseModel):
    name: str
    description: str
    type: str
    config: dict