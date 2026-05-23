"""Memory layer with FAISS vector store and embeddings."""

from .embeddings import EmbeddingClient
from .vector_store import VectorMemory, MemoryEntry

__all__ = ["EmbeddingClient", "VectorMemory", "MemoryEntry"]
