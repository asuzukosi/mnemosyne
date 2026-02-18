from src.core.search.base import BaseSearch
from src.core.data import QueryData, SearchResult, SearchConfig
from src.data_loader.base import BaseDataLoader
from src.core.index.base import BaseIndex
from sentence_transformers import SentenceTransformer


class SemanticSearch(BaseSearch):
    def __init__(self, config: SearchConfig, data_loader: BaseDataLoader, index: BaseIndex):
        self._config = config
        self._data_loader = data_loader
        self._index = index
        self._index.load()
        self._model = SentenceTransformer('all-MiniLM-L6-v2')
        self._model.max_seq_length = 256

    
    def search(self, query: QueryData, num_k:int=10) -> SearchResult:
        pass