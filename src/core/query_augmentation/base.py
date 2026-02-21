from abc import ABC, abstractmethod
from src.core.data import QueryData, QueryAugmentationConfig

class BaseQueryAugmentation(ABC):
    def __init__(self, config: QueryAugmentationConfig):
        self._config = config

    @abstractmethod
    def augment_query(self, query: QueryData) -> QueryData:
        raise NotImplementedError
    
