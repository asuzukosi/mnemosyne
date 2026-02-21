from src.core.evaluation.base import BaseEvaluation
from src.core.data import QueryData, SearchResult, EvaluationConfig
from typing import List

class Recall(BaseEvaluation):
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self._k = config.k

    def evaluate(self, query: QueryData, results: List[SearchResult]) -> float:
        scores = [self.score_result(query, result) for result in results]
        return sum(scores) / len(scores)
    
    def score_result(self, query: QueryData, result: SearchResult) -> float:
        return 1 if result.query == query.query else 0
    
    def score_results(self, query: QueryData, results: List[SearchResult]) -> List[float]:
        return [self.score_result(query, result) for result in results]