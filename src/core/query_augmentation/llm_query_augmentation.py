from src.core.query_augmentation.base import BaseQueryAugmentation
from src.core.data import QueryData, QueryAugmentationConfig
from src.clients.llm_client import LLMClient

class LLMQueryAugmentation(BaseQueryAugmentation):
    def __init__(self, config: QueryAugmentationConfig):
        super().__init__(config)
        self._llm_client = LLMClient(config=self._config)

    async def augment_query(self, query: QueryData) -> QueryData:
        augmented_query = await self._spelling_correction(query)
        augmented_query = await self._query_rewriting(augmented_query)
        return augmented_query
    
    async def _spelling_correction(self, query: QueryData) -> QueryData:
        system_prompt = "You are a helpful assistant that corrects the spelling of the user's query. return the corrected query only. do not return any other text or explanation."
        prompt = f"Correct the spelling of the following query: {query.query}"
        response: str = await self._llm_client.generate_response_with_context(prompt, system_prompt)
        return QueryData(query=response, context=response)
    
    async def _query_rewriting(self, query: QueryData) -> QueryData:
        system_prompt = "You are a helpful assistant that rewrites the user's query to be more specific and detailed. return the rewritten query only. do not return any other text or explanation."
        prompt = f"Rewrite the following query to be more specific and detailed: {query.query}"
        response: str = await self._llm_client.generate_response_with_context(prompt, system_prompt)
        return QueryData(query=response, context=response)
    