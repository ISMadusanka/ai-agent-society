"""
Reward functions for Multi-Agent RL.

Each function computes a scalar reward signal for a specific aspect
of agent behaviour.  The composite ``compute_reward`` function
aggregates them with configurable weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from utils.logger import SIM

if TYPE_CHECKING:
    from agents.agent import Agent, Action, ActionResult
    from society.relationships import RelationshipGraph
    from society.governance import SocietyState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual reward signals
# ---------------------------------------------------------------------------

def engagement_reward(action_result: "ActionResult") -> float:
    """Reward for generating engagement from other agents.

    More responses / reactions → higher reward.
    """
    n = len(action_result.responses)
    if n == 0:
        return 0.0
    # Diminishing returns: log-like scaling
    return min(1.0, 0.3 * n)


def alliance_reward(
    agent: "Agent",
    relationships: "RelationshipGraph",
) -> float:
    """Reward for maintaining positive alliances."""
    allies = relationships.get_allies(agent.agent_id, threshold=0.2)
    if not allies:
        return 0.0
    return min(1.0, 0.15 * len(allies))


def influence_reward(
    agent: "Agent",
    governance: "SocietyState",
) -> float:
    """Reward for having proposals accepted."""
    accepted = sum(
        1 for p in governance.proposals
        if p.proposer_id == agent.agent_id and p.status == "accepted"
    )
    return min(1.0, 0.25 * accepted)


def consistency_reward(
    agent: "Agent",
    action: "Action",
) -> float:
    """Reward for acting consistently with personality.

    High-extraversion agents are rewarded for speaking;
    high-conscientiousness agents for voting/proposing;
    high-agreeableness agents for cooperation/alliance.
    """
    traits = agent.personality.traits
    action_type = action.type.value

    score = 0.0
    if action_type in ("speak", "direct_message"):
        score = traits.get("extraversion", 0.5) * 0.3
    elif action_type in ("propose_role", "propose_rule", "vote"):
        score = traits.get("conscientiousness", 0.5) * 0.3
    elif action_type in ("form_alliance",):
        score = traits.get("agreeableness", 0.5) * 0.3
    elif action_type in ("challenge",):
        score = (1.0 - traits.get("agreeableness", 0.5)) * 0.2
    elif action_type == "reflect":
        score = traits.get("openness", 0.5) * 0.2
    elif action_type == "observe":
        score = (1.0 - traits.get("extraversion", 0.5)) * 0.1

    return score


def profile_development_reward(
    agent: "Agent",
    action: "Action",
) -> float:
    """Reward for actively defining one's identity.
    
    Provides a high reward when agents update their profile, 
    encouraging blank slate agents to establish themselves.
    """
    if action.type.value == "update_profile":
        return 0.8
    return 0.0


def social_harmony_reward(governance: "SocietyState") -> float:
    """Global reward signal based on society-wide metrics.

    Rewards the existence of structure (roles, rules) and
    active participation (proposals, votes).
    """
    role_count = len(governance.roles)
    rule_count = len([r for r in governance.rules if r.status == "active"])
    proposal_count = len(governance.proposals)

    # Normalised score (higher is better, capped at 1)
    structure = min(1.0, (role_count + rule_count) * 0.1)
    activity = min(1.0, proposal_count * 0.05)
    return (structure + activity) / 2.0


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """Configurable weights for each reward component."""

    engagement: float = 0.20
    alliance: float = 0.15
    influence: float = 0.15
    consistency: float = 0.15
    harmony: float = 0.15
    profile_development: float = 0.20


class RewardCalculator:
    """Aggregates individual reward signals into a single scalar."""

    def __init__(self, weights: Optional[RewardWeights] = None) -> None:
        self.weights = weights or RewardWeights()

    def compute(
        self,
        agent: "Agent",
        action: "Action",
        result: "ActionResult",
        relationships: "RelationshipGraph",
        governance: "SocietyState",
    ) -> float:
        """Compute the weighted composite reward for an agent's action."""
        w = self.weights

        r_engage = engagement_reward(result)
        r_alliance = alliance_reward(agent, relationships)
        r_influence = influence_reward(agent, governance)
        r_consist = consistency_reward(agent, action)
        r_harmony = social_harmony_reward(governance)
        r_profile = profile_development_reward(agent, action)

        total = (
            w.engagement * r_engage
            + w.alliance * r_alliance
            + w.influence * r_influence
            + w.consistency * r_consist
            + w.harmony * r_harmony
            + w.profile_development * r_profile
        )

        log.debug(
            f"{SIM} Reward for {agent.personality.name}: "
            f"total={total:.3f} "
            f"(engage={r_engage:.2f}, alliance={r_alliance:.2f}, "
            f"influence={r_influence:.2f}, consist={r_consist:.2f}, "
            f"harmony={r_harmony:.2f}, profile={r_profile:.2f})"
        )
        return total

