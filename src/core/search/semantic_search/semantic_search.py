from src.core.search.base import BaseSearch
from src.core.data import QueryData, SearchResult, SearchConfig
from src.data_loader.base import BaseDataLoader
from src.core.index.base import BaseIndex
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from typing import List
from src.data_loader.base import DataItem
from pathlib import Path
from src.utils.vector_ops import cosine_similarity

class SemanticSearch(BaseSearch):
    def __init__(self, config: SearchConfig, data_loader: BaseDataLoader, index: BaseIndex):
        self._config = config
        self._data_loader = data_loader
        self._index = index
        self._index.load()
        self._documents: List[DataItem] = []
        self._embeddings: List[np.ndarray] = []
        self._model = SentenceTransformer('all-MiniLM-L6-v2')
        self._model.max_seq_length = 256
        self._doc_map = defaultdict(str)
        self._embeddings_path = Path(__file__).parent / 'embeddings.npy'


    def generate_embeddings(self, text: str) -> np.ndarray:
        embeddings = self._model.encode([text])
        return embeddings[0]

    def embed_query(self, query: QueryData) -> np.ndarray:
        return self.generate_embeddings(query.query)

    def build_embeddings(self, documents: List[DataItem]) -> List[np.ndarray]:
        self._documents = documents
        self._embeddings = []
        self._doc_map = defaultdict(str)
        for document in documents:
            self._doc_map[document.id] = document.content
            self._embeddings.append(self.generate_embeddings(document.content))
        np.save(self._embeddings_path, self._embeddings)
        return self._embeddings
    
    def load_or_build_embeddings(self, documents: List[DataItem]) -> List[np.ndarray]:
        if self._embeddings_path.exists():
            self._embeddings = np.load(self._embeddings_path)
            self._documents = documents
            self._doc_map = defaultdict(str)
            for document in documents:
                self._doc_map[document.id] = document.content
                return self._embeddings
        else:
            return self.build_embeddings(documents)
    
    def search(self, query: QueryData, num_k:int=10) -> SearchResult:
        query_embedding = self.embed_query(query.query)
        scores = defaultdict(float)
        for embedding, doc_id in zip(self._embeddings, self._documents):
            scores[doc_id] = cosine_similarity(query_embedding, embedding)
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_k]
        return SearchResult(query=query.query, context=query.context, results=[self._doc_map[doc_id] for doc_id, _ in results])
        