from src.core.evaluation.base import BaseEvaluation
from src.core.data import QueryData, SearchResult, EvaluationConfig
from typing import List

class F1Score(BaseEvaluation):
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self._k = config.k

    def evaluate(self, query: QueryData, results: List[SearchResult]) -> float:
        return sum([self.score_result(query, result) for result in results]) / len(results)
    
    def score_result(self, query: QueryData, result: SearchResult) -> float:
        precision = self._get_precision(query, result)
        recall = self._get_recall(query, result)
        return self._get_f1_score(precision, recall)
    
    def _get_precision(self, query: QueryData, results: List[SearchResult]) -> float:
        return sum([self.score_result(query, result) for result in results]) / len(results)
    
    def _get_recall(self, query: QueryData, results: List[SearchResult]) -> float:
        return sum([self.score_result(query, result) for result in results]) / len(results)
    
    def _get_f1_score(self, precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall)