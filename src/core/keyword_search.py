from src.core.search.base import BaseSearch
from src.core.data import QueryData, SearchResult, SearchConfig

class KeywordSearch(BaseSearch):
    def __init__(self, config: SearchConfig):
        self._config = config

    def search(self, query: QueryData) -> SearchResult:
        pass