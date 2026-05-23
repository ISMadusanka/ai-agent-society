"""
Inter-agent message passing system.

Provides a ``MessageBus`` that manages broadcast and direct messages
between agents, with full history tracking per simulation step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from utils.logger import SOCIETY

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message data structure
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message between agents."""

    sender_id: str
    receiver_id: str  # Use "broadcast" for public messages
    content: str
    message_type: str = "speech"  # speech | proposal | vote | alliance | challenge
    timestamp: int = 0
    in_reply_to: Optional[str] = None  # sender_id of the message being replied to
    metadata: dict = field(default_factory=dict)

    @property
    def is_broadcast(self) -> bool:
        return self.receiver_id == "broadcast"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Message bus
# ---------------------------------------------------------------------------

class MessageBus:
    """Central message router for the agent society.

    Messages are queued per-step and delivered when agents poll
    for new input at the start of their turn.
    """

    def __init__(self) -> None:
        # Per-step message queue: agent_id → list of pending messages
        self._inbox: dict[str, list[Message]] = {}
        # Full history across all steps
        self.history: list[Message] = []
        # Broadcasts for the current step
        self._broadcasts: list[Message] = []

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send(self, message: Message) -> None:
        """Enqueue a message for delivery."""
        self.history.append(message)

        if message.is_broadcast:
            self._broadcasts.append(message)
            log.debug(
                f"{SOCIETY} 📢 {message.sender_id} broadcasts: "
                f"{message.content[:80]}…"
            )
        else:
            self._inbox.setdefault(message.receiver_id, []).append(message)
            log.debug(
                f"{SOCIETY} 💬 {message.sender_id} → {message.receiver_id}: "
                f"{message.content[:80]}…"
            )

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    def get_messages(self, agent_id: str) -> list[Message]:
        """Return all pending direct messages for *agent_id* (drains inbox)."""
        direct = self._inbox.pop(agent_id, [])
        return direct

    def get_broadcasts(self, exclude_sender: Optional[str] = None) -> list[Message]:
        """Return current-step broadcasts, optionally excluding a sender."""
        if exclude_sender:
            return [m for m in self._broadcasts if m.sender_id != exclude_sender]
        return list(self._broadcasts)

    def get_recent_history(self, n: int = 20) -> list[Message]:
        """Return the last *n* messages across all channels."""
        return list(self.history[-n:])

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def step_reset(self) -> None:
        """Clear per-step queues.  Called at the start of each sim step."""
        self._inbox.clear()
        self._broadcasts.clear()

    def to_dict_list(self) -> list[dict]:
        """Serialise full history for persistence."""
        return [m.to_dict() for m in self.history]

    @classmethod
    def from_dict_list(cls, data: list[dict]) -> "MessageBus":
        """Restore from serialised history."""
        bus = cls()
        bus.history = [Message(**item) for item in data]
        return bus
