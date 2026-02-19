from src.core.chunking.base import BaseChunker
from src.core.data import ChunkerConfig
from typing import List

class FixedSizedChunker(BaseChunker):
    def __init__(self, config: ChunkerConfig):
        super().__init__(config)

    def chunk(self, text: str, chunk_size: int) -> List[str]:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]