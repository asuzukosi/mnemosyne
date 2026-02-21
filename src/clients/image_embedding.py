from src.core.data import ImageEmbeddingConfig
from PIL import Image
import numpy as np
from transformers import AutoModel

class ImageEmbeddingClient:
    def __init__(self, config: ImageEmbeddingConfig):
       self._config = config

    def embed_image(self, image: Image) -> np.ndarray:
        raise NotImplementedError("ImageEmbeddingClient is not implemented")