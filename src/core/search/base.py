from src.core.data import QueryData, SearchResult, SearchConfig
from abc import ABC, abstractmethod
from src.data_loader.base import BaseDataLoader
from src.core.index.base import BaseIndex

class BaseSearch(ABC):
    def __init__(self, config: SearchConfig, data_loader: BaseDataLoader, index: BaseIndex):
        self._config = config
        self._data_loader = data_loader
        self._index = index
        
    @abstractmethod
    def search(self, query: QueryData, num_k:int=10) -> SearchResult:
        pass