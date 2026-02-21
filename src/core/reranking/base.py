from abc import ABC, abstractmethod
from src.core.data import QueryData, SearchResult, RerankerConfig
from typing import List

class BaseReranker(ABC):
    def __init__(self, config: RerankerConfig):
        self._config = config

    @abstractmethod
    def rerank(self, query: QueryData, results: List[SearchResult]) -> List[SearchResult]:
        raise NotImplementedError