"""
Agent reflection engine.

Periodically synthesises recent experiences into higher-level
insights ("reflections") that are stored back into memory.
This mimics the human ability to generalise from specific events
and improves long-term decision quality.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm.client import OllamaClient
from memory.vector_store import MemoryEntry
from utils.logger import AGENT

if TYPE_CHECKING:
    from agents.agent import Agent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REFLECT_SYSTEM = """\
You are helping an autonomous agent in a society simulation reflect on
its recent experiences.  Produce a thoughtful, first-person reflection
that captures lessons learned, patterns noticed, and updated beliefs.
Be concise (3-5 sentences).  Do NOT output JSON — just natural language."""

_REFLECT_PROMPT = """\
You are {name}.  Here is your personality:
{personality}

Your recent experiences (newest first):
{experiences}

Your current role in society: {role}

Based on these experiences, write a brief reflection.
What have you learned?  How do you feel about your relationships?
What should you focus on going forward?"""

_RELATIONSHIP_SYSTEM = """\
Summarise an agent's social standing in 2-3 sentences.
Be concise.  Do NOT output JSON."""

_RELATIONSHIP_PROMPT = """\
You are {name}.
Your relationships:
{relationships}

Recent interactions:
{recent_interactions}

Summarise your social standing and who you trust the most."""


class ReflectionEngine:
    """Produces reflections and relationship summaries for agents."""

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def reflect(
        self,
        agent: "Agent",
        recent_memories: list[MemoryEntry],
        step: int = 0,
    ) -> str:
        """Generate a reflection from recent experiences.

        The reflection is automatically stored in the agent's memory
        with type ``reflection`` and a small positive reward.
        """
        if not recent_memories:
            return ""

        experiences = "\n".join(
            f"- [{m.memory_type}] {m.text}" for m in recent_memories
        )
        prompt = _REFLECT_PROMPT.format(
            name=agent.personality.name,
            personality=agent.personality.summary(),
            experiences=experiences,
            role=agent.current_role,
        )

        try:
            result = self.llm.call(prompt, system=_REFLECT_SYSTEM)
            reflection = self.llm.get_response_text(result).strip()
        except Exception as exc:
            log.warning(f"{AGENT} Reflection failed for {agent.agent_id}: {exc}")
            reflection = "I need more time to process my experiences."

        # Store the reflection as a memory
        agent.memory.add(
            text=f"[Reflection] {reflection}",
            reward=0.2,  # Small positive reward for introspection
            memory_type="reflection",
            timestamp=step,
        )

        log.info(
            f"{AGENT} 🪞 {agent.personality.name} reflects: "
            f"{reflection[:100]}…"
        )
        return reflection

    def summarise_relationships(
        self,
        agent: "Agent",
        relationship_summary: str,
        recent_interactions: list[MemoryEntry],
    ) -> str:
        """Produce a social standing summary."""
        interactions_text = "\n".join(
            f"- {m.text}" for m in recent_interactions[:10]
        ) or "No recent interactions."

        prompt = _RELATIONSHIP_PROMPT.format(
            name=agent.personality.name,
            relationships=relationship_summary,
            recent_interactions=interactions_text,
        )

        try:
            result = self.llm.call(prompt, system=_RELATIONSHIP_SYSTEM)
            return self.llm.get_response_text(result).strip()
        except Exception:
            return "I am still getting to know everyone."
