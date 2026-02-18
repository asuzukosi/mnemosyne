from src.core.index.base import BaseIndex
from collections import defaultdict, Counter
from pathlib import Path
from typing import List
from src.core.data import DataItem
import math
import os
import pickle
from src.utils.logger import logger
from src.core.data import QueryData, SearchResult

BM25_K1 = 1.2

class InvertedIndex(BaseIndex):
    def __init__(self):
        self._index = defaultdict(set)
        self._docmap = {}
        self.index_path = Path(__file__).parent / 'index.pkl'
        self.docmap_path = Path(__file__).parent / 'docmap.pkl'
        self.term_frequency_path = Path(__file__).parent / 'term_frequency.pkl'
        self._term_frequency = defaultdict(Counter)

    def load(self):
        with open(self.index_path, 'rb') as f:
            self._index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self._docmap = pickle.load(f)
        with open(self.term_frequency_path, 'rb') as f:
            self._term_frequency = pickle.load(f)
        logger.info(f"Index loaded from {self.index_path}")
        logger.info(f"Docmap loaded from {self.docmap_path}")
        logger.info(f"Term frequency loaded from {self.term_frequency_path}")

    def get_documents(self, term: str) -> list[str]:
        return sorted(list(self._index[term]))
    
    def _tokenize_text(self, text: str) -> list[str]:
        return [token for token in text.split() if token.isalpha()]

    def get_idf(self, term: str) -> float:
        token = self._tokenize_text(term)[0]
        doc_count = len(self._docmap)
        term_doc_count = len(self._index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = self._tokenize_text(term)[0]
        doc_count = len(self._docmap)
        term_doc_count = len(self._index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: str, term: str) -> float:
        tf = self.get_term_frequency(doc_id, term)
        return tf * (BM25_K1 + 1) / (tf + BM25_K1)

    def _add_document(self, doc_id: str, text: str):
        tokens = self._tokenize_text(text)
        for token in set(tokens):
            self._index[token].add(doc_id)
        self._term_frequency[doc_id].update(tokens)

    def get_term_frequency(self, doc_id: str, term: str) -> Counter:
        token = self._tokenize_text(term)[0]
        return self._term_frequency[doc_id].get(token, 0) / len(self._term_frequency[doc_id].keys()) # normalize by the number of tokens in the document
    

    def get_tf_idf(self, doc_id: str, term: str) -> float:
        return self.get_term_frequency(doc_id, term) * self.get_idf(term)
    
    def build(self, data: List[DataItem]):
        for data_item in data:
            doc_id = data_item.id
            text = data_item.content # TODO: find another unique way of building the index
            self._add_document(doc_id, text)
            self._docmap[doc_id] = data.__dict__

    def save(self):
        # save index
        os.makedirs(self.index_path, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self._index, f)
        logger.info(f"Index saved to {self.index_path}")
        # save docmap
        os.makedirs(self.docmap_path, exist_ok=True)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self._docmap, f)
        logger.info(f"Docmap saved to {self.docmap_path}")
        # save term frequency
        os.makedirs(self.term_frequency_path, exist_ok=True)
        with open(self.term_frequency_path, 'wb') as f:
            pickle.dump(self._term_frequency, f)
        logger.info(f"Term frequency saved to {self.term_frequency_path}")


    def _search_index(self, query: QueryData, num_k:int=10) -> SearchResult: # TODO: why are we indexing?
        seen, res = set(), []
        query_tokens = self._tokenize_text(query.query)
        for token in query_tokens:
            matching_doc_ids = self.get_documents(token)
            for doc_id in matching_doc_ids:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                res.append(self._docmap[doc_id])
                if len(res) >= num_k:
                    break
        return SearchResult(query=query.query, context=query.context, results=res)