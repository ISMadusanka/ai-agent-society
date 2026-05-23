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

import os
from pathlib import Path

LOGS_DIR = Path("logs")
AGENTS_LOG_DIR = LOGS_DIR / "agents"

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
    os.makedirs(AGENTS_LOG_DIR, exist_ok=True)
    
    file_handler = logging.FileHandler("error.log", encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(file_formatter)

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
            ),
            file_handler
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Usage::

        log = get_logger(__name__)
        log.info(f"{AGENT} Agent-01 chose to SPEAK")
    """
    return logging.getLogger(name)

def get_agent_logger(agent_name: str) -> logging.Logger:
    """Return a dedicated file logger for an individual agent."""
    logger_name = f"agent_file.{agent_name}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(AGENTS_LOG_DIR / f"{agent_name}.log", encoding="utf-8")
        fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger

def get_society_logger() -> logging.Logger:
    """Return a dedicated file logger for society governance events."""
    logger = logging.getLogger("society_file")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOGS_DIR / "society.log", encoding="utf-8")
        fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger
