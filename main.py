"""
AI Agent Society — Entry Point

A multi-agent reinforcement learning simulation where autonomous
LLM-powered agents form and evolve their own society.

Usage::

    python main.py                         # 10 agents, 100 steps (defaults)
    python main.py --agents 5 --steps 50   # Customize
    python main.py --config my_config.yaml # Use custom config
    python main.py --resume                # Resume from saved state
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config
from simulation.engine import SimulationEngine
from utils.logger import setup_logging, get_logger, console

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Agent Society — Multi-Agent RL Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python main.py --agents 5 --steps 20
  python main.py --config config/default.yaml --model "gpt-oss:20b"
  python main.py --resume --steps 50
        """,
    )
    parser.add_argument(
        "--agents", type=int, default=None,
        help="Number of agents in the society (default: 10)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Number of simulation steps to run (default: 100)",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--ollama-url", type=str, default=None,
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ollama model name (default: gpt-oss:20b)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previously saved state",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    import logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    # Load configuration: YAML → CLI overrides
    config = Config.from_yaml(args.config)
    config = Config.from_args(args, base_config=config)

    # Banner
    console.print()
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]   🏛️  AI Agent Society  —  Multi-Agent RL[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print()
    console.print(f"  LLM Model    : [green]{config.llm.model}[/green]")
    console.print(f"  Ollama URL   : [green]{config.llm.base_url}[/green]")
    console.print(f"  Embed Model  : [green]{config.memory.embed_model}[/green]")
    console.print(f"  Agents       : [yellow]{config.simulation.num_agents}[/yellow]")
    console.print(f"  Steps        : [yellow]{config.simulation.max_steps}[/yellow]")
    console.print(f"  Persist Dir  : [dim]{config.simulation.persist_dir}[/dim]")
    console.print()

    # Create and run simulation
    engine = SimulationEngine(config)

    try:
        engine.initialise()
        engine.run()
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted — saving state…[/bold red]")
        engine._save_state()
    except ConnectionError as exc:
        console.print(f"\n[bold red]❌ {exc}[/bold red]")
        console.print(
            "[yellow]Make sure Ollama is running: "
            "ollama serve[/yellow]"
        )
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]❌ Unexpected error: {exc}[/bold red]")
        log.exception("Fatal error")
        sys.exit(1)

    console.print("\n[bold green]✓ Simulation complete![/bold green]")


if __name__ == "__main__":
    main()
