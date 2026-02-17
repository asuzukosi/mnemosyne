from src.core.search.base import BaseSearch
from src.core.data import QueryData, SearchResult, SearchConfig
from src.data_loader.base import BaseDataLoader

class KeywordSearch(BaseSearch):
    def __init__(self, config: SearchConfig, data_loader: BaseDataLoader):
        self._config = config
        self._data_loader = data_loader

    def search(self, query: QueryData, num_k:int=10) -> SearchResult:
        data = self._data_loader.load()
        results = []
        for item in data:
            if query.query.lower() == item.key.lower():
                results.append(item)
                # break when k items reached
                if len(results) >= num_k:
                    break
        return SearchResult(query=query.query, context=query.context, results=results)