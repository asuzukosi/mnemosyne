from src.core.evaluation.base import BaseEvaluation
from src.core.data import QueryData, SearchResult, EvaluationConfig
from src.clients.llm_client import LLMClient
from typing import List

class LLMEvaluation(BaseEvaluation):
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self._llm_client = LLMClient(config=self._config)

    async def evaluate(self, query: QueryData, results: List[SearchResult]) -> float:
        system_prompt = f"You are a helpful assistant that evaluates the search results based on the user's query. return the evaluation results only. do not return any other text or explanation."
        prompt = f"Evaluate the following search results based on the user's query: {query.query}. do not return any other text or explanation. the results are: {results}"
        response: str = await self._llm_client.generate_response_with_context(prompt, system_prompt)
        return response
    
    async def _individual_evaluate(self, query: QueryData, result: SearchResult) -> float:
        system_prompt = f"You are a helpful assistant that evaluates the search result based on the user's query. return the evaluation result only. do not return any other text or explanation."
        prompt = f"Evaluate the following search result based on the user's query: {query.query}. do not return any other text or explanation. the result is: {result}"
        response: str = await self._llm_client.generate_response_with_context(prompt, system_prompt)
        return response