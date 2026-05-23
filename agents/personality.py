"""
Agent personality generation.

Each agent receives a unique personality comprising Big Five traits,
core values, a background story, and a chosen name — all generated
by the LLM so that every simulation run produces a fresh cast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from llm.client import OllamaClient
from utils.logger import AGENT

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Personality data
# ---------------------------------------------------------------------------

@dataclass
class PersonalityProfile:
    """Complete personality description for a single agent."""

    name: str = "Unnamed"
    traits: dict = field(default_factory=lambda: {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
    })
    values: list[str] = field(default_factory=lambda: ["fairness"])
    background: str = ""
    speaking_style: str = "neutral"

    def summary(self) -> str:
        """One-paragraph summary suitable for prompt injection."""
        trait_desc = ", ".join(
            f"{k}={v:.1f}" for k, v in self.traits.items()
        )
        return (
            f"Name: {self.name}\n"
            f"Personality traits (0-1 scale): {trait_desc}\n"
            f"Core values: {', '.join(self.values)}\n"
            f"Background: {self.background}\n"
            f"Speaking style: {self.speaking_style}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "traits": self.traits,
            "values": self.values,
            "background": self.background,
            "speaking_style": self.speaking_style,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonalityProfile":
        return cls(**data)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_PERSONALITY_SYSTEM = """\
You are a character designer for a multi-agent society simulation.
Generate a unique, interesting personality for an autonomous agent.
Respond with ONLY valid JSON, no other text."""

_PERSONALITY_PROMPT = """\
Create a personality for Agent #{agent_id} in a society simulation.
The agent needs:
- A creative, memorable name (first name only, avoid common names)
- Big Five personality traits as decimals from 0.0 to 1.0
- 2-4 core values that guide their decisions
- A brief background story (2-3 sentences)
- A distinct speaking style (e.g., "formal and philosophical", "casual and humorous")

Each agent should feel like a distinct individual with clear motivations.
Previously created agents: {existing_names}

Return JSON in this exact format:
{{
  "name": "...",
  "traits": {{
    "openness": 0.0-1.0,
    "conscientiousness": 0.0-1.0,
    "extraversion": 0.0-1.0,
    "agreeableness": 0.0-1.0,
    "neuroticism": 0.0-1.0
  }},
  "values": ["value1", "value2"],
  "background": "...",
  "speaking_style": "..."
}}"""


def generate_personality(
    llm_client: OllamaClient,
    agent_id: str,
    existing_names: Optional[list[str]] = None,
) -> PersonalityProfile:
    """Use the LLM to generate a unique personality.

    Falls back to a deterministic default if the LLM call fails,
    ensuring the simulation can always start.
    """
    names_str = ", ".join(existing_names) if existing_names else "none yet"
    prompt = _PERSONALITY_PROMPT.format(
        agent_id=agent_id, existing_names=names_str
    )

    try:
        result = llm_client.call_json(prompt, system=_PERSONALITY_SYSTEM)

        profile = PersonalityProfile(
            name=result.get("name", f"Agent-{agent_id}"),
            traits=result.get("traits", PersonalityProfile().traits),
            values=result.get("values", ["fairness"]),
            background=result.get("background", "A newcomer to the society."),
            speaking_style=result.get("speaking_style", "neutral"),
        )
        log.info(f"{AGENT} Generated personality: {profile.name}")
        return profile

    except Exception as exc:
        log.warning(
            f"{AGENT} Personality generation failed for {agent_id}: {exc}. "
            f"Using fallback."
        )
        return _fallback_personality(agent_id)


def _fallback_personality(agent_id: str) -> PersonalityProfile:
    """Deterministic fallback when the LLM is unavailable."""
    import hashlib

    # Use agent_id hash to produce varied but repeatable traits
    h = int(hashlib.sha256(agent_id.encode()).hexdigest()[:8], 16)
    return PersonalityProfile(
        name=f"Agent-{agent_id}",
        traits={
            "openness": (h % 100) / 100,
            "conscientiousness": ((h >> 8) % 100) / 100,
            "extraversion": ((h >> 16) % 100) / 100,
            "agreeableness": ((h >> 24) % 100) / 100,
            "neuroticism": ((h >> 32) % 100) / 100,
        },
        values=["adaptability", "cooperation"],
        background="A newcomer figuring out their place in society.",
        speaking_style="straightforward",
    )
