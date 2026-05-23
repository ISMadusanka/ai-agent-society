"""Society structures: relationships, governance, and norms."""

from .relationships import RelationshipGraph, RelationshipEdge
from .governance import SocietyState, Proposal, Rule

__all__ = [
    "RelationshipGraph",
    "RelationshipEdge",
    "SocietyState",
    "Proposal",
    "Rule",
]
