from abc import ABC, abstractmethod
from src.core.data import DataItem, QueryData

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