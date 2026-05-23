"""
Weighted directed graph tracking agent-to-agent relationships.

Each edge stores *trust*, *familiarity*, and *sentiment* — three
orthogonal dimensions that evolve independently based on interaction
outcomes.  The graph is fully serialisable for state persistence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from utils.logger import SOCIETY

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge data
# ---------------------------------------------------------------------------

@dataclass
class RelationshipEdge:
    """Directed relationship from one agent to another."""

    from_id: str
    to_id: str
    trust: float = 0.0       # –1.0 (distrust) … +1.0 (full trust)
    familiarity: float = 0.0  # 0.0 (stranger) … 1.0 (well-known)
    sentiment: float = 0.0    # –1.0 (hostile) … +1.0 (warm)
    interaction_count: int = 0

    def strength(self) -> float:
        """Composite relationship strength (simple average)."""
        return (self.trust + self.familiarity + self.sentiment) / 3.0


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class RelationshipGraph:
    """Directed graph of agent relationships with decay and update logic."""

    def __init__(self) -> None:
        # (from_id, to_id) → RelationshipEdge
        self._edges: dict[tuple[str, str], RelationshipEdge] = {}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, from_id: str, to_id: str) -> RelationshipEdge:
        """Return the edge from *from_id* → *to_id*, creating if absent."""
        key = (from_id, to_id)
        if key not in self._edges:
            self._edges[key] = RelationshipEdge(from_id=from_id, to_id=to_id)
        return self._edges[key]

    def get_allies(self, agent_id: str, threshold: float = 0.3) -> list[str]:
        """Return IDs of agents with positive composite strength."""
        allies = []
        for (src, dst), edge in self._edges.items():
            if src == agent_id and edge.strength() >= threshold:
                allies.append(dst)
        return allies

    def get_all_for(self, agent_id: str) -> list[RelationshipEdge]:
        """Return every outgoing relationship edge for *agent_id*."""
        return [
            edge
            for (src, _), edge in self._edges.items()
            if src == agent_id
        ]

    def get_network_summary(self, agent_id: str) -> str:
        """Human-readable summary of an agent's social network."""
        edges = self.get_all_for(agent_id)
        if not edges:
            return "You have not interacted with anyone yet."

        lines = []
        for e in sorted(edges, key=lambda x: x.strength(), reverse=True):
            label = (
                "ally" if e.strength() > 0.3
                else "acquaintance" if e.strength() > 0
                else "rival"
            )
            lines.append(
                f"  - {e.to_id} ({label}): "
                f"trust={e.trust:+.2f}, "
                f"familiarity={e.familiarity:.2f}, "
                f"sentiment={e.sentiment:+.2f}"
            )
        return "Your relationships:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(
        self,
        from_id: str,
        to_id: str,
        interaction_type: str,
        positive: bool = True,
    ) -> RelationshipEdge:
        """Update an edge based on an interaction outcome.

        Parameters
        ----------
        interaction_type
            One of: ``speech``, ``alliance``, ``challenge``, ``vote``,
            ``cooperation``, ``conflict``.
        positive
            Whether the interaction had a positive outcome.
        """
        edge = self.get(from_id, to_id)
        edge.interaction_count += 1

        # Familiarity always grows (bounded at 1.0)
        edge.familiarity = min(1.0, edge.familiarity + 0.05)

        delta = 0.1 if positive else -0.1

        if interaction_type in ("alliance", "cooperation"):
            edge.trust = _clamp(edge.trust + delta * 1.5)
            edge.sentiment = _clamp(edge.sentiment + delta)
        elif interaction_type in ("challenge", "conflict"):
            edge.trust = _clamp(edge.trust + delta)
            edge.sentiment = _clamp(edge.sentiment + delta * 1.5)
        elif interaction_type == "vote":
            edge.trust = _clamp(edge.trust + delta * 0.5)
        else:  # speech, general interaction
            edge.trust = _clamp(edge.trust + delta * 0.3)
            edge.sentiment = _clamp(edge.sentiment + delta * 0.5)

        return edge

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> list[dict]:
        return [asdict(e) for e in self._edges.values()]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "RelationshipGraph":
        graph = cls()
        for item in data:
            edge = RelationshipEdge(**item)
            graph._edges[(edge.from_id, edge.to_id)] = edge
        return graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
