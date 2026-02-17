from src.core.search.base import BaseSearch
from src.core.data import QueryData, SearchResult, SearchConfig
from src.data_loader.base import BaseDataLoader
from typing import Set
from pathlib import Path
import string

class KeywordSearch(BaseSearch):
    def __init__(self, config: SearchConfig, data_loader: BaseDataLoader):
        self._config = config
        self._data_loader = data_loader
        self.stop_words_path = Path(__file__).parent / 'stop_words.txt'
        if not self.stop_words_path.exists():
            raise ValueError(f"File {self.stop_words_path} does not exist")
        if not self.stop_words_path.is_file():
            raise ValueError(f"File {self.stop_words_path} is not a file")
        self._stop_words = self.load_stop_words()

    def search(self, query: QueryData, num_k:int=10) -> SearchResult:
        data = self._data_loader.load()
        results = []
        for item in data:
            if self._compare_keys(query.query, item.key):
                results.append(item)
                # break when k items reached
                if len(results) >= num_k:
                    break
        return SearchResult(query=query.query, context=query.context, results=results)
    
    def load_stop_words(self) -> Set[str]:
        with open(self.stop_words_path, 'r') as f:
            return set([line.strip() for line in f.readlines()])
    

    def _tokenize_key(self, key: str) -> Set[str]:
        return set(key.split())
    
    def _clean_key(self, key: str) -> str:
        return key.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    
    def _compare_keys(self, key1: str, key2: str) -> bool:
        key1 = self._clean_key(key1)
        key2 = self._clean_key(key2)
        key1: Set[str] = self._tokenize_key(key1) - self._stop_words
        key2: Set[str] = self._tokenize_key(key2) - self._stop_words
        # check if there is any intersection between the two sets
        return len(key1.intersection(key2)) > 0 # TODO: we may need to check parts of the keys rather than exact values