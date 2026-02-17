from src.core.data import QueryData, SearchResult, SearchConfig
from abc import ABC, abstractmethod
from src.data_loader.base import BaseDataLoader

class BaseSearch(ABC):
    def __init__(self, config: SearchConfig, data_loader: BaseDataLoader):
        self._config = config

    @abstractmethod
    def search(self, query: QueryData, num_k:int=10) -> SearchResult:
        pass