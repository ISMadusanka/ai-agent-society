"""
Simulation engine — orchestrates the multi-agent society.

Drives the main loop:
  1. Deliver messages
  2. Each agent perceives the world
  3. Each agent selects an action (LLM call)
  4. Execute actions → update messages, relationships, governance
  5. Compute rewards → agents learn (store in FAISS)
  6. Periodic reflection & state persistence
  7. Rich dashboard update
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns

from config.settings import Config
from llm.client import OllamaClient
from memory.embeddings import EmbeddingClient
from agents.agent import Agent, Action, ActionType, ActionResult
from agents.reflection import ReflectionEngine
from rl.policy import LLMPolicy
from rl.rewards import RewardCalculator
from communication.message import Message, MessageBus
from society.relationships import RelationshipGraph
from society.governance import SocietyState
from utils.logger import SIM, AGENT, SOCIETY

log = logging.getLogger(__name__)


class SimulationEngine:
    """Main simulation loop for the AI Agent Society."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.console = Console()
        self.step = 0

        # --- Shared infrastructure ---
        self.llm = OllamaClient(
            base_url=config.llm.base_url,
            model=config.llm.model,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            temperature=config.llm.temperature,
        )
        self.embed_client = EmbeddingClient(
            model_name=config.memory.embed_model
        )
        self.policy = LLMPolicy(self.llm)
        self.reflection_engine = ReflectionEngine(self.llm)
        self.reward_calc = RewardCalculator()

        # --- Society structures ---
        self.message_bus = MessageBus()
        self.relationships = RelationshipGraph()
        self.governance = SocietyState()

        # --- Agents ---
        self.agents: list[Agent] = []

        # --- Persistence ---
        self.persist_dir = Path(config.simulation.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # --- Dashboard state ---
        self._recent_actions: list[str] = []
        self._step_rewards: dict[str, float] = {}

    # ==================================================================
    # Initialisation
    # ==================================================================

    def initialise(self) -> None:
        """Create agents and load any persisted state."""
        self.console.print(
            Panel(
                "[bold cyan]🏛️  AI Agent Society[/bold cyan]\n"
                f"Agents: {self.config.simulation.num_agents}  |  "
                f"Max steps: {self.config.simulation.max_steps}  |  "
                f"Model: {self.config.llm.model}",
                title="Initialising",
                border_style="cyan",
            )
        )

        # Try to load persisted state
        loaded = self._load_state()

        if not loaded:
            self._create_agents()
        else:
            self.console.print(
                f"[green]✓ Resumed from step {self.step} with "
                f"{len(self.agents)} agents[/green]"
            )

    def _create_agents(self) -> None:
        """Create fresh agents with LLM-generated personalities."""
        existing_names: list[str] = []
        mem_cfg = {
            "embed_dim": self.config.memory.embed_dim,
            "persist_dir": self.config.memory.persist_dir,
        }

        for i in range(self.config.simulation.num_agents):
            agent_id = f"agent-{i:02d}"
            agent = Agent(
                agent_id=agent_id,
                llm_client=self.llm,
                embed_client=self.embed_client,
                policy=self.policy,
                reflection_engine=self.reflection_engine,
                memory_config=mem_cfg,
            )

            self.console.print(
                f"  [cyan]Creating agent {i + 1}/"
                f"{self.config.simulation.num_agents}…[/cyan]",
                end=" ",
            )
            agent.init_personality(existing_names)
            existing_names.append(agent.personality.name)
            self.console.print(
                f"[green]→ {agent.personality.name}[/green]"
            )

            # Log the generated personality profile
            from utils.logger import get_agent_logger
            agent_log = get_agent_logger(agent.personality.name)
            agent_log.info("=" * 60)
            agent_log.info(f"AGENT BORN: {agent.personality.name} ({agent.agent_id})")
            agent_log.info(f"Traits: {agent.personality.traits}")
            agent_log.info(f"Values: {agent.personality.values}")
            agent_log.info(f"Background: {agent.personality.background}")
            agent_log.info(f"Speaking Style: {agent.personality.speaking_style}")
            agent_log.info("=" * 60)

            self.agents.append(agent)

        self.console.print(
            f"[bold green]✓ Created {len(self.agents)} agents[/bold green]"
        )

    # ==================================================================
    # Main simulation loop
    # ==================================================================

    def run(self) -> None:
        """Execute the simulation for ``max_steps`` steps."""
        max_steps = self.config.simulation.max_steps

        self.console.print(
            f"\n[bold yellow]▶ Starting simulation "
            f"(steps {self.step + 1} → {self.step + max_steps})[/bold yellow]\n"
        )

        try:
            with Live(
                self._build_dashboard(),
                console=self.console,
                refresh_per_second=1,
                transient=False,
            ) as live:
                for _ in range(max_steps):
                    self.step += 1
                    self._run_step()

                    # Update dashboard
                    live.update(self._build_dashboard())

                    # Periodic reflection
                    if self.step % self.config.simulation.reflection_interval == 0:
                        self._reflection_round()

                    # Resolve any pending proposals
                    self.governance.resolve_proposals(
                        total_agents=len(self.agents), step=self.step
                    )

                    # Periodic persistence
                    if self.step % self.config.simulation.persist_interval == 0:
                        self._save_state()

                    # Step delay
                    time.sleep(self.config.simulation.step_delay)

        except KeyboardInterrupt:
            self.console.print(
                "\n[bold red]⏹ Simulation interrupted by user[/bold red]"
            )
        finally:
            # Always save on exit
            self._save_state()
            self.console.print(
                f"[bold green]✓ State saved at step {self.step}[/bold green]"
            )

    # ==================================================================
    # Single step
    # ==================================================================

    def _run_step(self) -> None:
        """Execute one simulation step."""
        log.info(f"{SIM} ═══ Step {self.step} ═══")
        self._recent_actions.clear()
        self._step_rewards.clear()

        # Prepare shared context
        society_summary = self.governance.get_summary()
        pending = self._format_pending_proposals()
        recent_msgs = self._format_recent_messages()
        agent_list = [
            f"{a.personality.name} ({a.agent_id})" for a in self.agents
        ]

        # Reset per-step message queues
        self.message_bus.step_reset()

        # Each agent acts in sequence
        for agent in self.agents:
            # 1. Perceive
            broadcasts = self.message_bus.get_broadcasts(
                exclude_sender=agent.agent_id
            )
            direct = self.message_bus.get_messages(agent.agent_id)
            agent.perceive(broadcasts, direct, society_summary, self.step)

            # 2. Decide
            rel_summary = self.relationships.get_network_summary(agent.agent_id)
            other_agents = [
                f"{a.personality.name} ({a.agent_id})"
                for a in self.agents
                if a.agent_id != agent.agent_id
            ]
            action = agent.decide(
                recent_messages_text=recent_msgs,
                society_summary=society_summary,
                relationship_summary=rel_summary,
                pending_proposals=pending,
                agent_list=other_agents,
                step=self.step,
            )

            # 3. Execute
            result = self._execute_action(agent, action)

            # 4. Compute reward
            reward = self.reward_calc.compute(
                agent=agent,
                action=action,
                result=result,
                relationships=self.relationships,
                governance=self.governance,
            )

            # 5. Learn
            agent.learn(action, reward, self.step)

            # Log to dedicated agent file
            from utils.logger import get_agent_logger
            agent_log = get_agent_logger(agent.personality.name)
            agent_log.info(f"Step {self.step} | Action: {action.type.value}")
            agent_log.info(f"  Content: {action.content}")
            agent_log.info(f"  Target: {action.target_agent} | Reasoning: {action.reasoning}")
            agent_log.info(f"  Outcome: {result.description} | Reward: {reward:.2f}")
            agent_log.info("-" * 40)

            # Track for dashboard
            self._recent_actions.append(
                f"[{agent.personality.name}] {action.type.value}: "
                f"{action.content[:60]}"
            )
            self._step_rewards[agent.personality.name] = reward

    # ==================================================================
    # Action execution
    # ==================================================================

    def _execute_action(self, agent: Agent, action: Action) -> ActionResult:
        """Execute an agent's action and update the world state."""
        handlers = {
            ActionType.SPEAK: self._handle_speak,
            ActionType.DIRECT_MESSAGE: self._handle_direct_message,
            ActionType.PROPOSE_ROLE: self._handle_propose_role,
            ActionType.PROPOSE_RULE: self._handle_propose_rule,
            ActionType.VOTE: self._handle_vote,
            ActionType.FORM_ALLIANCE: self._handle_alliance,
            ActionType.REFLECT: self._handle_reflect,
            ActionType.UPDATE_PROFILE: self._handle_update_profile,
            ActionType.OBSERVE: self._handle_observe,
            ActionType.CHALLENGE: self._handle_challenge,
        }
        handler = handlers.get(action.type, self._handle_observe)
        return handler(agent, action)

    def _handle_speak(self, agent: Agent, action: Action) -> ActionResult:
        self.message_bus.send(Message(
            sender_id=agent.agent_id,
            receiver_id="broadcast",
            content=action.content,
            message_type="speech",
            timestamp=self.step,
        ))
        # Update relationships with all agents (familiarity)
        for other in self.agents:
            if other.agent_id != agent.agent_id:
                self.relationships.update(
                    agent.agent_id, other.agent_id, "speech", positive=True
                )
        return ActionResult(success=True, description="Spoke publicly")

    def _handle_direct_message(
        self, agent: Agent, action: Action
    ) -> ActionResult:
        target = self._resolve_target(action.target_agent)
        if not target:
            return ActionResult(
                success=False,
                description=f"Target agent not found: {action.target_agent}",
            )
        self.message_bus.send(Message(
            sender_id=agent.agent_id,
            receiver_id=target.agent_id,
            content=action.content,
            message_type="speech",
            timestamp=self.step,
        ))
        self.relationships.update(
            agent.agent_id, target.agent_id, "speech", positive=True
        )
        return ActionResult(
            success=True,
            description=f"Sent DM to {target.personality.name}",
            responses=[target.agent_id],
        )

    def _handle_propose_role(
        self, agent: Agent, action: Action
    ) -> ActionResult:
        target_id = None
        if action.target_agent:
            target = self._resolve_target(action.target_agent)
            target_id = target.agent_id if target else agent.agent_id
        else:
            target_id = agent.agent_id

        proposal = self.governance.propose(
            agent_id=agent.agent_id,
            proposal_type="role",
            content=action.content,
            target_agent=target_id,
            step=self.step,
        )
        # Broadcast the proposal
        self.message_bus.send(Message(
            sender_id=agent.agent_id,
            receiver_id="broadcast",
            content=f"I propose that {target_id} should be: {action.content}",
            message_type="proposal",
            timestamp=self.step,
            metadata={"proposal_id": proposal.id},
        ))
        return ActionResult(
            success=True, description=f"Proposed role: {action.content}"
        )

    def _handle_propose_rule(
        self, agent: Agent, action: Action
    ) -> ActionResult:
        proposal = self.governance.propose(
            agent_id=agent.agent_id,
            proposal_type="rule",
            content=action.content,
            step=self.step,
        )
        self.message_bus.send(Message(
            sender_id=agent.agent_id,
            receiver_id="broadcast",
            content=f"I propose a new rule: {action.content}",
            message_type="proposal",
            timestamp=self.step,
            metadata={"proposal_id": proposal.id},
        ))
        return ActionResult(
            success=True, description=f"Proposed rule: {action.content}"
        )

    def _handle_vote(self, agent: Agent, action: Action) -> ActionResult:
        # content should contain the proposal ID
        # Check for YES/FOR in reasoning or content
        proposal_id = action.content.strip()
        in_favour = True  # Default to yes

        # Try to parse vote direction from content
        lower_content = (action.content + " " + action.reasoning).lower()
        if any(w in lower_content for w in ["against", "no", "reject", "nay"]):
            in_favour = False

        # Try to find matching proposal
        pending = self.governance.get_pending_proposals()
        voted = False
        for prop in pending:
            if prop.id in proposal_id or proposal_id in prop.content[:30]:
                self.governance.vote(
                    agent.agent_id, prop.id, in_favour, self.step
                )
                self.relationships.update(
                    agent.agent_id, prop.proposer_id, "vote", positive=in_favour
                )
                voted = True
                break

        # If no specific proposal found, vote on the first pending one
        if not voted and pending:
            prop = pending[0]
            self.governance.vote(
                agent.agent_id, prop.id, in_favour, self.step
            )
            voted = True

        return ActionResult(
            success=voted,
            description=f"Voted {'FOR' if in_favour else 'AGAINST'}",
        )

    def _handle_alliance(self, agent: Agent, action: Action) -> ActionResult:
        target = self._resolve_target(action.target_agent)
        if not target:
            return ActionResult(
                success=False,
                description="Alliance target not found",
            )
        self.relationships.update(
            agent.agent_id, target.agent_id, "alliance", positive=True
        )
        self.relationships.update(
            target.agent_id, agent.agent_id, "alliance", positive=True
        )
        self.message_bus.send(Message(
            sender_id=agent.agent_id,
            receiver_id=target.agent_id,
            content=f"I'd like to form an alliance with you. {action.content}",
            message_type="alliance",
            timestamp=self.step,
        ))
        return ActionResult(
            success=True,
            description=f"Alliance proposed with {target.personality.name}",
            responses=[target.agent_id],
        )

    def _handle_reflect(self, agent: Agent, action: Action) -> ActionResult:
        reflection = agent.do_reflection(self.step)
        return ActionResult(
            success=True,
            description=f"Reflected: {reflection[:60]}",
        )

    def _handle_update_profile(self, agent: Agent, action: Action) -> ActionResult:
        agent.self_profile = action.content
        agent.memory.add(
            text=f"I updated my personal profile to:\n{action.content}",
            memory_type="observation",
            timestamp=self.step,
        )
        return ActionResult(
            success=True,
            description="Updated self-maintained profile",
        )

    def _handle_observe(self, agent: Agent, action: Action) -> ActionResult:
        agent.memory.add(
            text=f"[Observation] {action.content or 'Quietly observing.'}",
            memory_type="observation",
            timestamp=self.step,
        )
        return ActionResult(success=True, description="Observed quietly")

    def _handle_challenge(self, agent: Agent, action: Action) -> ActionResult:
        self.message_bus.send(Message(
            sender_id=agent.agent_id,
            receiver_id="broadcast",
            content=f"I challenge: {action.content}",
            message_type="challenge",
            timestamp=self.step,
        ))
        # Challenging can strain relationships
        if action.target_agent:
            target = self._resolve_target(action.target_agent)
            if target:
                self.relationships.update(
                    agent.agent_id, target.agent_id,
                    "challenge", positive=False,
                )
        return ActionResult(
            success=True,
            description=f"Challenged: {action.content[:60]}",
        )

    # ==================================================================
    # Reflection round
    # ==================================================================

    def _reflection_round(self) -> None:
        """Trigger reflection for all agents."""
        log.info(f"{SIM} 🪞 Reflection round at step {self.step}")
        for agent in self.agents:
            agent.do_reflection(self.step)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _resolve_target(self, target_str: Optional[str]) -> Optional[Agent]:
        """Find an agent by ID or name (fuzzy)."""
        if not target_str:
            return None
        target_lower = target_str.lower().strip()
        for agent in self.agents:
            if (
                agent.agent_id.lower() == target_lower
                or agent.personality.name.lower() == target_lower
                or target_lower in agent.agent_id.lower()
                or target_lower in agent.personality.name.lower()
            ):
                return agent
        return None

    def _format_recent_messages(self, n: int = 15) -> str:
        """Format recent message history for prompt injection."""
        msgs = self.message_bus.get_recent_history(n)
        if not msgs:
            return "No recent messages."
        lines = []
        for m in msgs:
            target = "everyone" if m.is_broadcast else m.receiver_id
            lines.append(f"  {m.sender_id} → {target}: {m.content[:100]}")
        return "\n".join(lines)

    def _format_pending_proposals(self) -> str:
        pending = self.governance.get_pending_proposals()
        if not pending:
            return ""
        lines = []
        for p in pending:
            lines.append(
                f"  [{p.id}] ({p.proposal_type}) \"{p.content}\" "
                f"by {p.proposer_id} — "
                f"votes: {p.votes_for}↑ {p.votes_against}↓"
            )
        return "\n".join(lines)

    # ==================================================================
    # Rich dashboard
    # ==================================================================

    def _build_dashboard(self) -> Layout:
        """Build a Rich layout showing the current simulation state."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=6),
        )

        # Header
        layout["header"].update(
            Panel(
                f"[bold cyan]🏛️  AI Agent Society[/bold cyan]  │  "
                f"Step: [yellow]{self.step}[/yellow]  │  "
                f"Agents: [green]{len(self.agents)}[/green]  │  "
                f"Rules: {len([r for r in self.governance.rules if r.status == 'active'])}  │  "
                f"Memories: {sum(a.memory.size for a in self.agents)}",
                border_style="cyan",
            )
        )

        # Body: agents table + recent actions
        layout["body"].split_row(
            Layout(name="agents", ratio=1),
            Layout(name="actions", ratio=1),
        )

        # Agent status table
        agent_table = Table(
            title="Agent Status",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )
        agent_table.add_column("Name", style="cyan", width=14)
        agent_table.add_column("Role", style="green", width=12)
        agent_table.add_column("Mood", width=10)
        agent_table.add_column("Energy", width=8)
        agent_table.add_column("Memories", width=9)
        agent_table.add_column("Reward", width=8)

        for agent in self.agents:
            mood_emoji = {
                "positive": "😊",
                "content": "🙂",
                "neutral": "😐",
                "frustrated": "😤",
            }.get(agent.mood, "😐")

            reward = self._step_rewards.get(agent.personality.name, 0.0)
            reward_style = "green" if reward > 0.3 else "yellow" if reward > 0.1 else "red"

            agent_table.add_row(
                agent.personality.name,
                agent.current_role,
                f"{mood_emoji} {agent.mood}",
                f"{'█' * int(agent.energy * 5)}{'░' * (5 - int(agent.energy * 5))}",
                str(agent.memory.size),
                f"[{reward_style}]{reward:.2f}[/{reward_style}]",
            )

        layout["agents"].update(Panel(agent_table, border_style="dim"))

        # Recent actions
        actions_text = "\n".join(
            self._recent_actions[-12:]
        ) or "[dim]Waiting for actions…[/dim]"
        layout["actions"].update(
            Panel(
                actions_text,
                title="Recent Actions",
                border_style="dim",
            )
        )

        # Footer: society summary
        society_info = []
        if self.governance.roles:
            roles_str = ", ".join(
                f"{aid}: {role}" for aid, role in list(self.governance.roles.items())[:5]
            )
            society_info.append(f"[green]Roles:[/green] {roles_str}")
        active_rules = [r for r in self.governance.rules if r.status == "active"]
        if active_rules:
            rules_str = " | ".join(r.description[:40] for r in active_rules[:3])
            society_info.append(f"[yellow]Rules:[/yellow] {rules_str}")
        pending = self.governance.get_pending_proposals()
        if pending:
            society_info.append(
                f"[magenta]Pending proposals:[/magenta] {len(pending)}"
            )
        if not society_info:
            society_info.append("[dim]Society is forming…[/dim]")

        layout["footer"].update(
            Panel(
                "\n".join(society_info),
                title="Society",
                border_style="green",
            )
        )

        return layout

    # ==================================================================
    # Persistence
    # ==================================================================

    def _save_state(self) -> None:
        """Save full simulation state to disk."""
        log.info(f"{SIM} 💾 Saving state at step {self.step}…")

        state = {
            "step": self.step,
            "agents": [a.get_state_dict() for a in self.agents],
            "governance": self.governance.to_dict(),
            "relationships": self.relationships.to_dict(),
            "messages": self.message_bus.to_dict_list(),
        }

        state_path = self.persist_dir / "simulation_state.json"
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

        # Save each agent's FAISS index
        for agent in self.agents:
            agent.memory.save()

        log.info(f"{SIM} ✓ State saved ({state_path})")

    def _load_state(self) -> bool:
        """Load simulation state from disk.  Returns True if loaded."""
        state_path = self.persist_dir / "simulation_state.json"
        if not state_path.exists():
            return False

        self.console.print("[yellow]Found saved state, loading…[/yellow]")

        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)

        self.step = state.get("step", 0)
        self.governance = SocietyState.from_dict(state.get("governance", {}))
        self.relationships = RelationshipGraph.from_dict(
            state.get("relationships", [])
        )
        self.message_bus = MessageBus.from_dict_list(
            state.get("messages", [])
        )

        # Recreate agents with persisted personalities and state
        mem_cfg = {
            "embed_dim": self.config.memory.embed_dim,
            "persist_dir": self.config.memory.persist_dir,
        }
        for agent_data in state.get("agents", []):
            agent = Agent(
                agent_id=agent_data["agent_id"],
                llm_client=self.llm,
                embed_client=self.embed_client,
                policy=self.policy,
                reflection_engine=self.reflection_engine,
                memory_config=mem_cfg,
            )
            agent.load_state_dict(agent_data)
            agent.memory.load()
            self.agents.append(agent)

            # Log resumption
            from utils.logger import get_agent_logger
            agent_log = get_agent_logger(agent.personality.name)
            agent_log.info("=" * 60)
            agent_log.info(f"SIMULATION RESUMED (Step {self.step})")
            if agent.self_profile:
                agent_log.info(f"Current Self-Profile:\n{agent.self_profile}")
            agent_log.info("=" * 60)

        return True
