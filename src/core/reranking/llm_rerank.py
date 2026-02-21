from src.core.reranking.base import BaseReranker
from src.core.data import QueryData, SearchResult, RerankerConfig
from src.clients.llm_client import LLMClient
from typing import List

class LLMRerank(BaseReranker):
    def __init__(self, config: RerankerConfig):
        super().__init__(config)
        self._llm_client = LLMClient(config=self._config)

    async def rerank(self, query: QueryData, results: List[SearchResult]) -> List[SearchResult]:
        reranked_results = []
        for result in results:
            reranked_result = await self._individual_rerank(query, result)
            reranked_results.append(reranked_result)
        return reranked_results
    

    async def _individual_rerank(self, query: QueryData, result: SearchResult) -> SearchResult:
        result_context = "\n".join([f"Result {i+1}: {result.results[i].content}" for i in range(len(result.results))])
        system_prompt = f"You are a helpful assistant that reranks the search results based on the user's query. return the reranked results only"
        prompt = f"Rerank the following search results based on the user's query: {query.query}. do not return any other text or explanation. the results are: {result_context}"
        response: str = await self._llm_client.generate_response_with_context(prompt, system_prompt)
        return SearchResult(query=query.query, context=query.context, results=response.split("\n"))