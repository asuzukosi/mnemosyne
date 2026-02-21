from src.core.data import QueryData, LLMConfig
from typing import List
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion


class LLMClient:
    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = AsyncOpenAI(api_key=self._config.api_key, base_url=self._config.base_url)


    async def generate_response(self, message: str) -> str:
        messages = [{"role": "user", "content": message}]
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._config.model_name,
            messages=messages,
            temperature=0.0,
            stream=False,
        )
        return response.choices[0].message.content
    
    async def generate_response_with_context(self, message: str, context: str) -> str:
        messages = [ {"role": "system", "content": context}, {"role": "user", "content": message},]
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._config.model_name,
            messages=messages,
            temperature=0.0,
            stream=False,
        )
        return response.choices[0].message.content