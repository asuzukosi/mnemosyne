from src.core.reranking.base import BaseReranker
from src.core.data import QueryData, SearchResult, RerankerConfig
from sentence_transformers import CrossEncoder
from typing import List

class CrossEncoderRerank(BaseReranker):
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    async def rerank(self, query: QueryData, results: List[SearchResult]) -> List[SearchResult]:
        reranked_results = []
        for result in results:
            reranked_result = await self._individual_rerank(query, result)
            reranked_results.append(reranked_result)
        return reranked_results
    
    async def _individual_rerank(self, query: QueryData, result: SearchResult) -> SearchResult:
        scores = self._cross_encoder.predict([[query.query, result.results[0].content]])
        return SearchResult(query=query.query, context=query.context, results=[result.results[0]], score=scores[0])