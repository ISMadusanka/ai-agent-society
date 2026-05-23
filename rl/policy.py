"""
LLM-based policy for action selection.

The policy constructs rich prompts that include the agent's personality,
retrieved memories (reward-weighted), recent history, and available
actions.  The LLM output is parsed into a structured ``Action``.
"""

from __future__ import annotations

import json
import logging
import random
from typing import TYPE_CHECKING, Optional

from llm.client import OllamaClient
from memory.vector_store import MemoryEntry
from utils.logger import AGENT

if TYPE_CHECKING:
    from agents.agent import Agent, Action, ActionType

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_POLICY_SYSTEM = """\
You are an autonomous agent in a society simulation.  You must choose
an action from the available options and provide content for that action.

CRITICAL: Respond with ONLY valid JSON in this exact format:
{{
  "action": "<action_type>",
  "content": "<what you want to say or propose>",
  "target_agent": "<agent_id or null>",
  "reasoning": "<brief internal reasoning>"
}}

Available action types:
  speak          - Say something to all nearby agents
  direct_message - Send a private message to a specific agent
  propose_role   - Propose a role for yourself or another agent
  propose_rule   - Suggest a new rule or norm for the society
  vote           - Vote on a pending proposal (content: proposal_id, target_agent: null)
  form_alliance  - Propose an alliance with another agent
  reflect        - Spend this turn reflecting on your experiences
  observe        - Quietly observe without acting
  challenge      - Challenge an existing rule or role"""


def _build_action_prompt(
    agent: "Agent",
    memories: list[MemoryEntry],
    recent_messages: str,
    society_summary: str,
    relationship_summary: str,
    pending_proposals: str,
    agent_list: list[str],
) -> str:
    """Construct the full decision-making prompt."""
    memory_text = "\n".join(
        f"  - [{m.memory_type}] (reward={m.reward:.2f}) {m.text}"
        for m in memories
    ) or "  No relevant memories."

    return f"""\
You are {agent.personality.name} (ID: {agent.agent_id}).

== YOUR PERSONALITY ==
{agent.personality.summary()}

== YOUR CURRENT STATE ==
Role: {agent.current_role}
Energy: {agent.energy:.1f}/1.0
Mood: {agent.mood}
Goals: {', '.join(agent.goals) if agent.goals else 'None set'}

== YOUR RELEVANT MEMORIES ==
{memory_text}

== RECENT MESSAGES IN SOCIETY ==
{recent_messages or 'No recent messages.'}

== SOCIETY STATE ==
{society_summary}

== YOUR RELATIONSHIPS ==
{relationship_summary}

== PENDING PROPOSALS (vote with proposal ID) ==
{pending_proposals or 'No pending proposals.'}

== OTHER AGENTS ==
{', '.join(agent_list)}

Based on your personality, memories, and the current situation,
choose your next action.  Be true to your character.
If there are pending proposals, consider voting.
If you have ideas for the society, propose rules or roles.
Interact with other agents — address them by name."""


class LLMPolicy:
    """Selects agent actions using the LLM with RL-informed context."""

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def select_action(
        self,
        agent: "Agent",
        memories: list[MemoryEntry],
        recent_messages: str,
        society_summary: str,
        relationship_summary: str,
        pending_proposals: str,
        agent_list: list[str],
    ) -> dict:
        """Query the LLM and return a parsed action dict.

        Returns a dict with keys: ``action``, ``content``,
        ``target_agent``, ``reasoning``.
        """
        prompt = _build_action_prompt(
            agent=agent,
            memories=memories,
            recent_messages=recent_messages,
            society_summary=society_summary,
            relationship_summary=relationship_summary,
            pending_proposals=pending_proposals,
            agent_list=agent_list,
        )

        # Temperature variation based on mood/energy for exploration
        temp = self._compute_temperature(agent)

        try:
            result = self.llm.call_json(
                prompt, system=_POLICY_SYSTEM, temperature=temp
            )
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {"raw": str(result)}

            # Validate required fields
            if "action" not in result or not isinstance(result["action"], str):
                result["action"] = "observe"
            if "content" not in result or not isinstance(result["content"], str):
                result["content"] = ""
            result.setdefault("target_agent", None)
            result.setdefault("reasoning", "")

            # Log safely
            content_str = str(result.get('content', ''))
            log.info(
                f"{AGENT} 🎯 {agent.personality.name} decides: "
                f"{result['action']} → {content_str[:60]}"
            )
            return result

        except Exception as exc:
            log.exception(
                f"{AGENT} Policy failed for {agent.personality.name}: {exc}. "
                f"Falling back to observe."
            )
            return {
                "action": "observe",
                "content": "Looking around thoughtfully.",
                "target_agent": None,
                "reasoning": "Policy fallback due to error.",
            }

    def _compute_temperature(self, agent: "Agent") -> float:
        """Vary LLM temperature based on agent state (exploration)."""
        base = 0.7
        # Low energy → more conservative (lower temp)
        energy_mod = (agent.energy - 0.5) * 0.2
        # High neuroticism → more erratic (higher temp)
        neuro = agent.personality.traits.get("neuroticism", 0.5)
        neuro_mod = (neuro - 0.5) * 0.15
        # Small random jitter
        jitter = random.uniform(-0.05, 0.05)

        temp = base + energy_mod + neuro_mod + jitter
        return max(0.3, min(1.2, temp))
