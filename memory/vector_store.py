"""
FAISS-based vector memory store — one instance per agent.

Each agent maintains a private FAISS index that stores embedded
experiences alongside structured metadata.  Retrieval uses
**reward-weighted similarity**: raw cosine similarity is scaled
by a configurable factor of the stored reward, so high-reward
memories surface more readily — the core RL learning signal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from memory.embeddings import EmbeddingClient
from utils.logger import AGENT

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory record stored alongside its FAISS vector."""

    text: str
    timestamp: int = 0
    reward: float = 0.0
    memory_type: str = "observation"  # observation | interaction | reflection | decision | social_feedback
    related_agent: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vector memory
# ---------------------------------------------------------------------------

class VectorMemory:
    """Per-agent FAISS vector store with reward-weighted retrieval."""

    def __init__(
        self,
        agent_id: str,
        embed_client: EmbeddingClient,
        dim: int = 384,
        persist_dir: str = "data/memories",
    ) -> None:
        self.agent_id = agent_id
        self.embed_client = embed_client
        self.dim = dim
        self.persist_dir = Path(persist_dir) / agent_id
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Inner product on L2-normalised vectors == cosine similarity
        cpu_index = faiss.IndexFlatIP(dim)
        
        # Try to move index to GPU if available
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            log.debug(f"{AGENT} Initialised FAISS index on GPU for {agent_id}")
        except Exception as e:
            self.index = cpu_index
            log.debug(f"{AGENT} Initialised FAISS index on CPU for {agent_id} (GPU not available: {e})")
            
        self.entries: list[MemoryEntry] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        reward: float = 0.0,
        memory_type: str = "observation",
        related_agent: Optional[str] = None,
        timestamp: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Embed *text* and store it with its metadata."""
        vec = self.embed_client.embed(text).reshape(1, -1)
        self.index.add(vec)
        self.entries.append(
            MemoryEntry(
                text=text,
                timestamp=timestamp,
                reward=reward,
                memory_type=memory_type,
                related_agent=related_agent,
                metadata=metadata or {},
            )
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        reward_weight: float = 0.3,
    ) -> list[MemoryEntry]:
        """Retrieve the most relevant memories via reward-weighted cosine.

        Scoring:  ``final_score = cosine_sim × (1 + α × reward)``

        where *α* is ``reward_weight``.  This causes high-reward
        experiences to rank higher for the same semantic similarity.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = self.embed_client.embed(query).reshape(1, -1)
        # Fetch extra candidates for re-ranking
        k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        ranked: list[tuple[float, MemoryEntry]] = []
        for raw_score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.entries[idx]
            weighted = float(raw_score) * (1.0 + reward_weight * entry.reward)
            ranked.append((weighted, entry))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in ranked[:top_k]]

    def get_recent(self, n: int = 5) -> list[MemoryEntry]:
        """Return the *n* most recently added memories."""
        return list(reversed(self.entries[-n:]))

    @property
    def size(self) -> int:
        return self.index.ntotal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        try:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        except AttributeError:
            cpu_index = self.index
        except Exception:
            cpu_index = self.index
            
        faiss.write_index(cpu_index, str(self.persist_dir / "index.faiss"))
        meta = [asdict(e) for e in self.entries]
        with open(self.persist_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        log.debug(
            f"{AGENT} Saved {self.index.ntotal} memories for {self.agent_id}"
        )

    def load(self) -> bool:
        """Restore from disk.  Returns ``True`` if data was loaded."""
        index_path = self.persist_dir / "index.faiss"
        meta_path = self.persist_dir / "metadata.json"

        if not (index_path.exists() and meta_path.exists()):
            return False

        cpu_index = faiss.read_index(str(index_path))
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception:
            self.index = cpu_index
            
        with open(meta_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.entries = [MemoryEntry(**item) for item in raw]
        log.info(
            f"{AGENT} Loaded {self.index.ntotal} memories for {self.agent_id}"
        )
        return True
