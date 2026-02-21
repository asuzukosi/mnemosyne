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
    
    def weighted_search(self, query: QueryData, num_k:int=10) -> SearchResult:
        raise NotImplementedError
    
    def rrf_search(self, query: QueryData, num_k:int=10) -> SearchResult:
        raise NotImplementedError
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return [1.0 for _ in scores]
        range_score = max_score - min_score
        return [(score - min_score) / range_score for score in scores]