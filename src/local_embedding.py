# src/local_embedding.py
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class LocalEmbeddingPipeline:
    """
    Pipeline de embeddings 100% local usando sentence-transformers.
    Por defecto: all-MiniLM-L6-v2 (384 dims, CPU-friendly).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> list[list[float]]:
        """
        Devuelve una lista de vectores (list[float]) de dimensión 384.
        """
        if not texts:
            return []

        # np.array de shape (n_chunks, 384)
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Lo convertimos a listas para que psycopg/pgvector lo acepte bien
        return emb.tolist()

    def embed_text(self, text: str) -> list[float]:
        """
        Embedding de una sola cadena.
        """
        if not text:
            return [0.0] * 384
        return self.embed_texts([text])[0]

    @staticmethod
    def average_embeddings(embeddings: list[list[float]]) -> list[float]:
        """
        Media de varios vectores (mismo tamaño).
        """
        if not embeddings:
            return []
        arr = np.array(embeddings, dtype=float)
        avg = arr.mean(axis=0)
        return avg.tolist()
