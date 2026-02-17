from src.core.data import QueryData, SearchResult, SearchConfig
from abc import ABC, abstractmethod

class BaseSearch(ABC):
    def __init__(self, config: SearchConfig):
        self._config = config

    @abstractmethod
    def search(self, query: QueryData) -> SearchResult:
        pass