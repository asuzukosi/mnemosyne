from abc import ABC, abstractmethod
from src.core.data import DataItem, QueryData
from src.core.data import SearchResult

class BaseIndex(ABC):
    
    def get_documents(self, term: str) -> str:
        raise NotImplementedError
    
    def load(self):
        raise NotImplementedError

    def _add_document(self, doc_id: str, text: str):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
    
    def index_based_search(self, query: QueryData, num_k:int=10) -> SearchResult:
        raise NotImplementedError