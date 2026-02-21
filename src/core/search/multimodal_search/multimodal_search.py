from src.core.search.base import BaseSearch
from src.core.data import QueryData, SearchResult, SearchConfig
from src.clients.llm_client import LLMClient
from src.core.index.base import BaseIndex
from typing import List

class MultimodalSearch(BaseSearch):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self._llm_client = LLMClient(config=self._config)
        self._index = BaseIndex(config=self._config)

    async def search(self, query: QueryData) -> List[SearchResult]:
        query_description = await self._describe_query(query)
        search_results = await self._index.index_based_search(query_description)
        return search_results

    async def _describe_query(self, query: QueryData) -> str:
        system_prompt = f"You are a helpful assistant that describes the user's query in a way that is easy to understand. return the description only. do not return any other text or explanation."
        prompt = f"Describe the following user's query: {query.query}. do not return any other text or explanation."
        response: str = await self._llm_client.generate_response_with_context(prompt, system_prompt)
        return response