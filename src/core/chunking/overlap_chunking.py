
from src.core.chunking.base import BaseChunker
from src.core.data import ChunkerConfig
from typing import List

class OverlapChunker(BaseChunker):
    def __init__(self, config: ChunkerConfig):
        super().__init__(config)

    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = [] # TODO: this chunking is inaccurate and should be improved
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if i > 0:
                chunk = chunk + text[i-overlap:i]
            if i + chunk_size < len(text):
                chunk = chunk + text[i+chunk_size:i+chunk_size+overlap]
            chunks.append(chunk)
        return chunks