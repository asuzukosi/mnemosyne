from typing import List
from src.core.data import ChunkerConfig
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    def __init__(self, config: ChunkerConfig):
        self._config = config

    @abstractmethod
    def chunk(self, text: str, chunk_size: int) -> List[str]:
        pass