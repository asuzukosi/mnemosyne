from src.core.chunking.base import BaseChunker
from src.core.data import ChunkerConfig
from typing import List
import re

class SemanticChunker(BaseChunker):
    def __init__(self, config: ChunkerConfig):
        super().__init__(config)

    def chunk(self, text: str, chunk_size: int) -> List[str]:
        sentences = self._split_sentences(text)
        chunks = []
        chunk_sentences = ""
        for sentence in sentences:
            chunk_sentences += sentence
            if len(chunk_sentences) >= chunk_size:
                chunks.append(chunk_sentences)
                chunk_sentences = ""
        if chunk_sentences:
            chunks.append(chunk_sentences)
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r"(?<=[.!?])\s+", text)