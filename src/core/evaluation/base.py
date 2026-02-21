from abc import ABC, abstractmethod
from src.core.data import QueryData, SearchResult, EvaluationConfig
from typing import List

class BaseEvaluation(ABC):
    def __init__(self, config: EvaluationConfig):
        self._config = config

    @abstractmethod
    def evaluate(self, query: QueryData, results: List[SearchResult]) -> float:
        raise NotImplementedError