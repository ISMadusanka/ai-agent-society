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

def generate_personality(
    llm_client: OllamaClient,
    agent_id: str,
    existing_names: Optional[list[str]] = None,
) -> PersonalityProfile:
    """Generate a blank slate personality profile.
    
    Agents start with neutral traits and empty backgrounds, allowing them
    to dynamically construct their identity through interactions.
    """
    profile = PersonalityProfile(
        name=f"Agent-{agent_id}",
        traits={
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        },
        values=[],
        background="",
        speaking_style="neutral",
    )
    log.info(f"{AGENT} Initialised blank slate personality for: {profile.name}")
    return profile
