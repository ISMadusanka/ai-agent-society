"""
Structured logging with Rich console output.

Provides colour-coded, categorised logging for the simulation:
  - AGENT   → cyan      (agent actions & decisions)
  - LLM     → magenta   (API calls & responses)
  - SOCIETY → green     (governance & relationship events)
  - SIM     → yellow    (simulation engine lifecycle)
"""

import logging
from rich.logging import RichHandler
from rich.console import Console

# Shared Rich console (reused by the dashboard as well)
console = Console()

# ---------------------------------------------------------------------------
# Category tags used in log messages
# ---------------------------------------------------------------------------
AGENT = "[bold cyan][AGENT][/bold cyan]"
LLM = "[bold magenta][LLM][/bold magenta]"
SOCIETY = "[bold green][SOCIETY][/bold green]"
SIM = "[bold yellow][SIM][/bold yellow]"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with Rich output.

    Call once at application startup.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Usage::

        log = get_logger(__name__)
        log.info(f"{AGENT} Agent-01 chose to SPEAK")
    """
    return logging.getLogger(name)
