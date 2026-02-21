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
    chunk_size: int

class LLMConfig(BaseModel):
    name: str
    model_name: str
    api_key: str
    base_url: str
    description: str
    type: str
    config: dict

class DataLoaderConfig(BaseModel):
    name: str
    description: str
    type: str
    config: dict

class ChunkerConfig(BaseModel):
    name: str
    description: str
    type: str
    config: dict


class QueryAugmentationConfig(BaseModel):
    name: str
    description: str
    type: str
    config: dict