"""
Configuration management for AI Agent Society.

Loads settings from YAML files with CLI override support.
Uses dataclasses for type-safe, documented configuration.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Ollama LLM connection settings."""

    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss:20b"
    timeout: int = 120
    max_retries: int = 3
    temperature: float = 0.8


@dataclass
class MemoryConfig:
    """FAISS vector memory settings."""

    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384
    top_k: int = 5
    reward_weight: float = 0.3
    persist_dir: str = "data/memories"


@dataclass
class SimulationConfig:
    """Simulation engine settings."""

    num_agents: int = 10
    max_steps: int = 100
    step_delay: float = 0.5
    reflection_interval: int = 5
    persist_interval: int = 10
    persist_dir: str = "data/state"
    max_concurrent_calls: int = 5


@dataclass
class Config:
    """Root configuration container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.

        Missing keys fall back to dataclass defaults, so a partial
        YAML is perfectly valid.
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        return cls(
            llm=LLMConfig(**{**asdict(LLMConfig()), **raw.get("llm", {})}),
            memory=MemoryConfig(**{**asdict(MemoryConfig()), **raw.get("memory", {})}),
            simulation=SimulationConfig(
                **{**asdict(SimulationConfig()), **raw.get("simulation", {})}
            ),
        )

    @classmethod
    def from_args(cls, args, base_config: Optional["Config"] = None) -> "Config":
        """Override a base config with CLI arguments.

        Only non-``None`` argument values are applied so that the
        YAML / default values are preserved for unset flags.
        """
        cfg = base_config or cls()

        if getattr(args, "agents", None) is not None:
            cfg.simulation.num_agents = args.agents
        if getattr(args, "steps", None) is not None:
            cfg.simulation.max_steps = args.steps
        if getattr(args, "ollama_url", None) is not None:
            cfg.llm.base_url = args.ollama_url
        if getattr(args, "model", None) is not None:
            cfg.llm.model = args.model

        return cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the entire config tree to a plain dict."""
        return asdict(self)
