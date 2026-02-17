from src.data_loader.base import BaseDataLoader
from src.core.data import DataItem, DataLoaderConfig
import json
from pathlib import Path
import uuid

class JSONLoader(BaseDataLoader):
    def __init__(self, config: DataLoaderConfig, path: Path):
        self._path = path
        if not self._path.exists():
            raise FileNotFoundError(f"File {self._path} does not exist")
        if not self._path.is_file():
            raise NotADirectoryError(f"File {self._path} is not a file")
        super().__init__(config)

    def load(self) -> list[DataItem]:
        with open(self._path, 'r') as f:
            data = json.load(f)
        return [DataItem(id=item['id'] if 'id' in item else str(uuid.uuid4()), 
                         key=item['key'], 
                         content=item['content'], 
                         metadata=item) for item in data]