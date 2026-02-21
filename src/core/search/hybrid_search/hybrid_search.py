from src.core.search.base import BaseSearch
from src.core.data import QueryData, SearchResult, SearchConfig
from src.data_loader.base import BaseDataLoader
from src.core.index.base import BaseIndex
from typing import List

class HybridSearch(BaseSearch):
    def __init__(self, config: SearchConfig, data_loader: BaseDataLoader, index: BaseIndex):
        self._config = config
        self._data_loader = data_loader
        self._index = index
        self._index.build()
        self._index.save()

    def _bm_25_search(self, query: QueryData, num_k:int=10) -> SearchResult:
        return self._index.index_based_search(query, num_k)
    
    def search(self, query: QueryData, num_k:int=10) -> SearchResult:
        return self._bm_25_search(query, num_k)
    
    def _semantic_search(self, query: QueryData, num_k:int=10) -> SearchResult:
        return self._index.index_based_search(query, num_k)
    
    def weighted_search(self, query: QueryData, num_k:int=10) -> SearchResult:
        bm25_results = self._bm_25_search(query, num_k * 100)
        semantic_results = self._semantic_search(query, num_k * 100)
        weighted_results = self.hybrid_score(self.normalize_scores(bm25_results), self.normalize_scores(semantic_results))
        return weighted_results
    
    def combine_results(self, bm25_results: List[SearchResult], semantic_results: List[SearchResult]) -> List[SearchResult]:
        # TODO: combine the scores from both the bm25 scores and the semantic scores
        raise NotImplementedError
    
    def _rrf_combine_results(self, bm25_results: List[SearchResult], semantic_results: List[SearchResult]) -> List[SearchResult]:
        # TODO: combine the scores from both the bm25 scores and the semantic scores
        raise NotImplementedError
    
    def rrf_search(self, query: QueryData, num_k:int=10) -> SearchResult:
        bm25_results = self._bm_25_search(query, num_k * 100)
        semantic_results = self._semantic_search(query, num_k * 100)
        combined_results = self._rrf_combine_results(bm25_results, semantic_results)
        weighted_results = self.hybrid_score(self.normalize_scores(combined_results), self.normalize_scores(semantic_results))
        return weighted_results
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return [1.0 for _ in scores]
        range_score = max_score - min_score
        return [(score - min_score) / range_score for score in scores]
    

    def hybrid_score(self, bm25_score: List[float], semantic_score: List[float], alpha: float=0.5) -> List[float]:
        return [alpha * bm25_score[i] + (1 - alpha) * semantic_score[i] for i in range(len(bm25_score))]