from abc import ABC, abstractmethod
from src.core.data import DataItem, DataLoaderConfig

class BaseDataLoader(ABC):
    def __init__(self, config: DataLoaderConfig):
        self._config = config

    @abstractmethod
    def load(self) -> list[DataItem]:
        pass