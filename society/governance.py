"""
Society governance: roles, rules, proposals, and voting.

Tracks the emergent social structure that agents create through
proposals and voting.  Everything is fully serialisable for
cross-session persistence.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, asdict, field
from typing import Optional

from utils.logger import SOCIETY

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A societal norm or rule enacted by the agents."""

    id: str
    proposer_id: str
    description: str
    status: str = "active"  # active | repealed
    enacted_at: int = 0


@dataclass
class Proposal:
    """A pending proposal awaiting votes."""

    id: str
    proposer_id: str
    proposal_type: str  # "role" | "rule" | "repeal"
    content: str
    target_agent: Optional[str] = None  # For role proposals
    votes: dict = field(default_factory=dict)  # agent_id → True/False
    status: str = "pending"  # pending | accepted | rejected
    created_at: int = 0

    @property
    def votes_for(self) -> int:
        return sum(1 for v in self.votes.values() if v)

    @property
    def votes_against(self) -> int:
        return sum(1 for v in self.votes.values() if not v)


@dataclass
class GovernanceEvent:
    """Immutable record of a governance action."""

    event_type: str  # proposal_created | vote_cast | proposal_resolved | role_assigned
    agent_id: str
    description: str
    step: int = 0


# ---------------------------------------------------------------------------
# Society state
# ---------------------------------------------------------------------------

class SocietyState:
    """Central tracker for the society's self-organising structure."""

    def __init__(self) -> None:
        self.roles: dict[str, str] = {}  # agent_id → role name
        self.rules: list[Rule] = []
        self.proposals: list[Proposal] = []
        self.history: list[GovernanceEvent] = []

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

    def assign_role(self, agent_id: str, role: str, step: int = 0) -> None:
        old = self.roles.get(agent_id)
        self.roles[agent_id] = role
        self.history.append(
            GovernanceEvent(
                event_type="role_assigned",
                agent_id=agent_id,
                description=f"{agent_id} assigned role '{role}' (was '{old}')",
                step=step,
            )
        )
        log.info(f"{SOCIETY} 👑 {agent_id} → role: {role}")

    def get_role(self, agent_id: str) -> str:
        return self.roles.get(agent_id, "citizen")

    def get_agents_with_role(self, role: str) -> list[str]:
        return [aid for aid, r in self.roles.items() if r == role]

    # ------------------------------------------------------------------
    # Proposals
    # ------------------------------------------------------------------

    def propose(
        self,
        agent_id: str,
        proposal_type: str,
        content: str,
        target_agent: Optional[str] = None,
        step: int = 0,
    ) -> Proposal:
        """Create a new proposal for the society to vote on."""
        proposal = Proposal(
            id=str(uuid.uuid4())[:8],
            proposer_id=agent_id,
            proposal_type=proposal_type,
            content=content,
            target_agent=target_agent,
            created_at=step,
        )
        self.proposals.append(proposal)
        self.history.append(
            GovernanceEvent(
                event_type="proposal_created",
                agent_id=agent_id,
                description=f"Proposed ({proposal_type}): {content}",
                step=step,
            )
        )
        log.info(f"{SOCIETY} 📋 {agent_id} proposes: {content[:80]}")
        return proposal

    def vote(
        self,
        agent_id: str,
        proposal_id: str,
        in_favour: bool,
        step: int = 0,
    ) -> bool:
        """Cast a vote on a pending proposal.  Returns True if found."""
        for proposal in self.proposals:
            if proposal.id == proposal_id and proposal.status == "pending":
                proposal.votes[agent_id] = in_favour
                self.history.append(
                    GovernanceEvent(
                        event_type="vote_cast",
                        agent_id=agent_id,
                        description=(
                            f"Voted {'FOR' if in_favour else 'AGAINST'} "
                            f"proposal {proposal_id}"
                        ),
                        step=step,
                    )
                )
                return True
        return False

    def get_pending_proposals(self) -> list[Proposal]:
        return [p for p in self.proposals if p.status == "pending"]

    def resolve_proposals(self, total_agents: int, step: int = 0) -> list[Proposal]:
        """Resolve proposals that have received enough votes.

        A simple majority among voters is required, and at least
        30 % of agents must have voted.
        """
        resolved = []
        quorum = max(2, int(total_agents * 0.3))

        for proposal in self.proposals:
            if proposal.status != "pending":
                continue
            total_votes = len(proposal.votes)
            if total_votes < quorum:
                continue

            accepted = proposal.votes_for > proposal.votes_against
            proposal.status = "accepted" if accepted else "rejected"

            self.history.append(
                GovernanceEvent(
                    event_type="proposal_resolved",
                    agent_id=proposal.proposer_id,
                    description=(
                        f"Proposal {proposal.id} {proposal.status}: "
                        f"{proposal.content[:60]} "
                        f"({proposal.votes_for}–{proposal.votes_against})"
                    ),
                    step=step,
                )
            )
            log.info(
                f"{SOCIETY} ✅ Proposal {proposal.status}: "
                f"{proposal.content[:60]} "
                f"({proposal.votes_for}–{proposal.votes_against})"
            )

            # Apply accepted proposals
            if accepted:
                self._apply_proposal(proposal, step)

            resolved.append(proposal)

        return resolved

    def _apply_proposal(self, proposal: Proposal, step: int) -> None:
        """Side-effect handler for accepted proposals."""
        if proposal.proposal_type == "role" and proposal.target_agent:
            self.assign_role(proposal.target_agent, proposal.content, step)
        elif proposal.proposal_type == "rule":
            self.rules.append(
                Rule(
                    id=str(uuid.uuid4())[:8],
                    proposer_id=proposal.proposer_id,
                    description=proposal.content,
                    enacted_at=step,
                )
            )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """Human-readable summary for injection into agent prompts."""
        lines = ["Current Society State:"]

        # Roles
        if self.roles:
            lines.append("  Roles:")
            for aid, role in self.roles.items():
                lines.append(f"    - {aid}: {role}")
        else:
            lines.append("  Roles: None assigned yet")

        # Active rules
        active_rules = [r for r in self.rules if r.status == "active"]
        if active_rules:
            lines.append("  Active Rules:")
            for rule in active_rules:
                lines.append(f"    - {rule.description} (by {rule.proposer_id})")
        else:
            lines.append("  Active Rules: None")

        # Pending proposals
        pending = self.get_pending_proposals()
        if pending:
            lines.append(f"  Pending Proposals ({len(pending)}):")
            for p in pending:
                lines.append(
                    f"    - [{p.id}] ({p.proposal_type}) {p.content[:50]} "
                    f"by {p.proposer_id} "
                    f"(votes: {p.votes_for}↑ {p.votes_against}↓)"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "roles": self.roles,
            "rules": [asdict(r) for r in self.rules],
            "proposals": [asdict(p) for p in self.proposals],
            "history": [asdict(e) for e in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SocietyState":
        state = cls()
        state.roles = data.get("roles", {})
        state.rules = [Rule(**r) for r in data.get("rules", [])]
        state.proposals = [Proposal(**p) for p in data.get("proposals", [])]
        state.history = [GovernanceEvent(**e) for e in data.get("history", [])]
        return state
