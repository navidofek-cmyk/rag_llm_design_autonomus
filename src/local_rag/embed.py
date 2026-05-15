import os

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
    ) -> None:
        self.model_name = os.environ.get("EMBED_MODEL", model_name)
        device = os.environ.get("EMBED_DEVICE", device)
        self.model = SentenceTransformer(self.model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_one(self, text: str) -> np.ndarray:
        result = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return result[0]
