"""
Sentence Transformer embedding client.

Wraps a local SentenceTransformer model to produce normalised float32
vectors suitable for FAISS ``IndexFlatIP`` (cosine similarity).

The model is loaded lazily on first use and includes an in-process
cache to avoid redundant re-embeddings within a single run.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logger import LLM

log = logging.getLogger(__name__)


class EmbeddingClient:
    """Produces dense vector embeddings using a local SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log.info(f"{LLM} Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            log.info(
                f"{LLM} Embedding model loaded "
                f"(dim={self._model.get_sentence_embedding_dimension()})"
            )
        return self._model

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string, returning a 1-D float32 vector.

        Results are cached in-process to avoid redundant computation.
        """
        if text in self._cache:
            return self._cache[text]

        vec = self.model.encode(
            text, normalize_embeddings=True
        ).astype(np.float32)

        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings, returning a 2-D float32 matrix.

        Each row is a normalised embedding vector.
        """
        return self.model.encode(
            texts, normalize_embeddings=True
        ).astype(np.float32)

    def clear_cache(self) -> None:
        """Free the in-process embedding cache."""
        self._cache.clear()
