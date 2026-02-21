from src.core.search.base import BaseSearch
from src.core.data import QueryData, SearchResult, SearchConfig
from src.clients.llm_client import LLMClient
from src.core.index.base import BaseIndex
from typing import List
import numpy as np
from src.clients.image_embedding import ImageEmbeddingClient
from sentence_transformers import SentenceTransformer
from PIL import Image

class MultimodalSearch(BaseSearch):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self._llm_client = LLMClient(config=self._config)
        self._index = BaseIndex(config=self._config)
        self._text_embedder = SentenceTransformer('clip-ViT-B-32')
        self._image_embedder = ImageEmbeddingClient(config=self._config)

    async def search(self, query: QueryData) -> List[SearchResult]:
        query_description = self._embed_query(query.query)
        search_results = await self._index.index_based_search(query_description)
        # TODO: compare the embeddings to the embeddings of the iamges
        return search_results

    def _embed_query(self, text: str) -> np.ndarray:
        return self._text_embedder.encode([text])

    async def _embed_images(self, images: str) -> List[np.ndarray]:
        images = [Image.open(image) for image in images]
        return self._image_embedder.embed_image(images)