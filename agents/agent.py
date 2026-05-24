"""
Core Agent class — the autonomous entity in the society.

Each agent has:
  - A unique personality (LLM-generated)
  - Private FAISS memory (reward-weighted retrieval)
  - Internal state (energy, mood, goals, role)
  - A lifecycle: perceive → decide → act → learn
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from llm.client import OllamaClient
from memory.embeddings import EmbeddingClient
from memory.vector_store import VectorMemory, MemoryEntry
from agents.personality import PersonalityProfile, generate_personality
from agents.reflection import ReflectionEngine
from rl.policy import LLMPolicy
from communication.message import Message
from utils.logger import AGENT

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(Enum):
    SPEAK = "speak"
    DIRECT_MESSAGE = "direct_message"
    PROPOSE_ROLE = "propose_role"
    PROPOSE_RULE = "propose_rule"
    VOTE = "vote"
    FORM_ALLIANCE = "form_alliance"
    REFLECT = "reflect"
    OBSERVE = "observe"
    CHALLENGE = "challenge"
    UPDATE_PROFILE = "update_profile"


@dataclass
class Action:
    """A structured action chosen by an agent."""

    type: ActionType
    content: str
    target_agent: Optional[str] = None
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ActionResult:
    """Outcome of executing an action."""

    success: bool
    description: str
    responses: list = field(default_factory=list)
    reward: float = 0.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """An autonomous agent in the society simulation."""

    def __init__(
        self,
        agent_id: str,
        llm_client: OllamaClient,
        embed_client: EmbeddingClient,
        policy: LLMPolicy,
        reflection_engine: ReflectionEngine,
        memory_config: Optional[dict] = None,
    ) -> None:
        self.agent_id = agent_id
        self.llm = llm_client
        self.policy = policy
        self.reflection_engine = reflection_engine

        # Memory
        mem_cfg = memory_config or {}
        self.memory = VectorMemory(
            agent_id=agent_id,
            embed_client=embed_client,
            dim=mem_cfg.get("embed_dim", 384),
            persist_dir=mem_cfg.get("persist_dir", "data/memories"),
        )

        # Personality (generated later via init_personality)
        self.personality = PersonalityProfile()

        # Internal state
        self.current_role: str = "citizen"
        self.energy: float = 1.0
        self.mood: str = "neutral"
        self.goals: list[str] = []
        self.self_profile: str = ""

        # Per-step accumulators
        self._pending_messages: list[Message] = []
        self._current_perceptions: list[str] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_personality(self) -> None:
        """Initialise a blank slate personality."""
        self.personality = generate_personality(
            self.llm, self.agent_id
        )

    def try_load_state(self) -> bool:
        """Attempt to restore previous state from disk.

        Returns ``True`` if state was loaded successfully.
        """
        return self.memory.load()

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def perceive(
        self,
        broadcasts: list[Message],
        direct_messages: list[Message],
        society_summary: str,
        step: int = 0,
    ) -> list[str]:
        """Process incoming information and build perceptions.

        Stores incoming messages as memories so they influence
        future retrieval.
        """
        perceptions: list[str] = []

        for msg in broadcasts:
            text = f"{msg.sender_id} said publicly: {msg.content}"
            perceptions.append(text)
            self.memory.add(
                text=text,
                memory_type="observation",
                related_agent=msg.sender_id,
                timestamp=step,
            )

        for msg in direct_messages:
            text = f"{msg.sender_id} told you: {msg.content}"
            perceptions.append(text)
            self.memory.add(
                text=text,
                memory_type="interaction",
                related_agent=msg.sender_id,
                timestamp=step,
            )

        self._current_perceptions = perceptions
        self._pending_messages = list(direct_messages)
        return perceptions

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------

    def decide(
        self,
        recent_messages_text: str,
        society_summary: str,
        relationship_summary: str,
        pending_proposals: str,
        agent_list: list[str],
        step: int = 0,
    ) -> Action:
        """Select an action using the RL policy (LLM + reward-weighted memory).

        This is the core RL loop: the policy uses memories ranked by
        reward to construct context, so high-reward experiences
        directly influence future decisions.
        """
        # Build a context query from current perceptions
        context_query = " ".join(self._current_perceptions[-5:])
        if not context_query:
            context_query = f"society step {step} {society_summary[:100]}"

        # Retrieve reward-weighted memories
        memories = self.memory.retrieve(context_query, top_k=5)

        # Query the LLM policy
        raw = self.policy.select_action(
            agent=self,
            memories=memories,
            recent_messages=recent_messages_text,
            society_summary=society_summary,
            relationship_summary=relationship_summary,
            pending_proposals=pending_proposals,
            agent_list=agent_list,
        )

        # Parse into structured Action
        action = self._parse_action(raw)
        return action

    def _parse_action(self, raw: dict) -> Action:
        """Convert LLM JSON output into a typed Action."""
        action_str = raw.get("action", "observe").lower().strip()

        # Map string to enum (with fallback)
        try:
            action_type = ActionType(action_str)
        except ValueError:
            log.debug(
                f"{AGENT} Unknown action '{action_str}' from "
                f"{self.personality.name}, defaulting to observe"
            )
            action_type = ActionType.OBSERVE

        return Action(
            type=action_type,
            content=raw.get("content", ""),
            target_agent=raw.get("target_agent"),
            reasoning=raw.get("reasoning", ""),
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, action: Action, reward: float, step: int = 0) -> None:
        """Store the action and its reward in memory.

        This is the RL feedback loop: future retrievals will
        preferentially surface high-reward memories, shaping
        subsequent decisions.
        """
        text = (
            f"[Action: {action.type.value}] {action.content}"
            f"{' → ' + action.target_agent if action.target_agent else ''}"
        )
        self.memory.add(
            text=text,
            reward=reward,
            memory_type="decision",
            related_agent=action.target_agent,
            timestamp=step,
        )

        # Update internal state based on reward
        self._update_state(action, reward)

    def _update_state(self, action: Action, reward: float) -> None:
        """Adjust energy and mood based on action outcome."""
        # Energy cost per action type
        costs = {
            ActionType.SPEAK: 0.05,
            ActionType.DIRECT_MESSAGE: 0.03,
            ActionType.PROPOSE_ROLE: 0.08,
            ActionType.PROPOSE_RULE: 0.08,
            ActionType.VOTE: 0.02,
            ActionType.FORM_ALLIANCE: 0.06,
            ActionType.REFLECT: 0.04,
            ActionType.OBSERVE: 0.01,
            ActionType.CHALLENGE: 0.10,
            ActionType.UPDATE_PROFILE: 0.05,
        }
        self.energy = max(0.0, self.energy - costs.get(action.type, 0.05))

        # Natural energy recovery
        self.energy = min(1.0, self.energy + 0.03)

        # Mood adjustment
        if reward > 0.5:
            self.mood = "positive"
        elif reward > 0.2:
            self.mood = "content"
        elif reward > 0.0:
            self.mood = "neutral"
        else:
            self.mood = "frustrated"

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    def do_reflection(self, step: int = 0) -> str:
        """Trigger a reflection cycle on recent experiences."""
        recent = self.memory.get_recent(n=10)
        return self.reflection_engine.reflect(self, recent, step)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Serialise agent state (excluding FAISS which is saved separately)."""
        return {
            "agent_id": self.agent_id,
            "personality": self.personality.to_dict(),
            "current_role": self.current_role,
            "energy": self.energy,
            "mood": self.mood,
            "goals": self.goals,
            "self_profile": self.self_profile,
        }

    def load_state_dict(self, data: dict) -> None:
        """Restore agent state from a dict."""
        self.personality = PersonalityProfile.from_dict(data["personality"])
        self.current_role = data.get("current_role", "citizen")
        self.energy = data.get("energy", 1.0)
        self.mood = data.get("mood", "neutral")
        self.goals = data.get("goals", [])
        self.self_profile = data.get("self_profile", "")

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id}, name={self.personality.name}, "
            f"role={self.current_role}, energy={self.energy:.1f}, "
            f"mood={self.mood})"
        )
